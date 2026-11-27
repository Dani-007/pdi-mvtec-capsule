"""Etapa 7: atributos da região suspeita e SVM / Random Forest sobre a saída dos especialistas,
comparados com a regra da cascata (normal x defeituosa, tipo e sistema completo).

Uso:  python src/classificacao_cascata.py [--recalcular]   (~10 min, depois de src/cascata.py)
"""

from __future__ import annotations

import argparse
import csv

import cv2
import numpy as np

import cascata
from atributos import descrever_forma, estatisticas_histograma, lbp
from cascata import ALFA, DIR_MASCARAS, DIR_RESULT, TIPOS, _chave, carregar_valores, decidir, mapear
from cascata import AZUL, FUNDO, GRADE, LARANJA, TINTA, TINTA_2
from especialistas import (
    DILATA_IMPRESSAO, ESTAGIOS, NOMES, _ee, diferenca_do_500, preparar, referencias_do_banco,
)
from particao import carregar_particao
from pipeline import RAIZ, carregar_imagem

CSV_ATRIBUTOS = RAIZ / "dados" / "atributos_cascata.csv"

# Anel de 7 px em volta da região, para o contraste com a vizinhança (mais largo que a borda
#   borrada de um defeito, 2-4 px, e estreito o bastante para não pegar outra estrutura).
LARGURA_ANEL = 7
GLCM_NIVEIS = 16         # tons de cinza reduzidos a 16 níveis (matriz 16 x 16)
HOG_FAIXAS = 8           # orientações do gradiente em 8 faixas de 22,5 graus (0-180)
SEMENTE = 42
CONFIG_SISTEMA = ("completo", "RandomForest")   # ficha e classificador do "sistema final"

GRUPOS = {
    "valores": list(ESTAGIOS),
    "geometria": ["n_manchas", "area_total", "largura_caixa", "altura_caixa", "area_maior",
                  "perimetro", "compacidade", "excentricidade", "extensao", "solidez",
                  "eixo_maior", "orientacao"],
    "textura": ["media_cinza", "desvio_cinza", "contraste_anel",
                "media_a", "media_b", "desvio_a", "desvio_b",
                "suavidade", "assimetria", "uniformidade", "entropia",
                "lbp_entropia", "lbp_uniformidade",
                "glcm_contraste", "glcm_homogeneidade", "glcm_energia", "glcm_correlacao"]
               + [f"hog_{i}" for i in range(HOG_FAIXAS)],
    "posicao": ["frac_tampa_preta", "frac_sobre_500", "dist_contorno", "posicao_x"],
    "diferenca": ["dif500_area", "dif500_frac_clara", "dif500_claro_fora", "dif500_escuro_tinta"],
}
COR = ["media_a", "media_b", "desvio_a", "desvio_b"]
GLCM = ["glcm_contraste", "glcm_homogeneidade", "glcm_energia", "glcm_correlacao"]
HOG = [f"hog_{i}" for i in range(HOG_FAIXAS)]
GLOBAL = ["g_" + a for a in GRUPOS["textura"] if a != "contraste_anel"]
COMPLETO = [a for g in GRUPOS.values() for a in g]
ATRIBUTOS = COMPLETO + GLOBAL


# --- 1. DESCRITORES ---
def glcm(cinza: np.ndarray, mascara: np.ndarray, niveis: int = GLCM_NIVEIS) -> dict[str, float]:
    """GLCM (vizinhos a 1 px, 4 direções, só dentro da máscara): contraste, homogeneidade,
    energia e correlação."""
    q = (cinza.astype(np.int32) * niveis) // 256
    dentro = mascara > 0
    h, w = q.shape
    p = np.zeros((niveis, niveis))
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        xa, xb = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        ya, yb = slice(0, h - dy), slice(dy, h)
        par = dentro[ya, xa] & dentro[yb, xb]
        np.add.at(p, (q[ya, xa][par], q[yb, xb][par]), 1)
    p = p + p.T
    if p.sum() == 0:
        return {"glcm_contraste": 0.0, "glcm_homogeneidade": 0.0, "glcm_energia": 0.0,
                "glcm_correlacao": 0.0}
    p /= p.sum()
    i, j = np.indices(p.shape)
    mi, mj = np.sum(i * p), np.sum(j * p)
    si, sj = np.sqrt(np.sum((i - mi) ** 2 * p)), np.sqrt(np.sum((j - mj) ** 2 * p))
    return {"glcm_contraste": float(np.sum((i - j) ** 2 * p)),
            "glcm_homogeneidade": float(np.sum(p / (1 + (i - j) ** 2))),
            "glcm_energia": float(np.sum(p ** 2)),
            "glcm_correlacao": float(np.sum((i - mi) * (j - mj) * p) / (si * sj))
            if si > 0 and sj > 0 else 1.0}


def hog(cinza: np.ndarray, mascara: np.ndarray, faixas: int = HOG_FAIXAS) -> dict[str, float]:
    """HOG de uma célula só: histograma das orientações do gradiente (Sobel), ponderado pela
    magnitude e normalizado."""
    gx = cv2.Sobel(cinza, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(cinza, cv2.CV_32F, 0, 1, ksize=3)
    dentro = mascara > 0
    mag = np.hypot(gx, gy)[dentro]
    ang = (np.degrees(np.arctan2(gy, gx))[dentro]) % 180
    hist = np.bincount((ang * faixas / 180).astype(int) % faixas, weights=mag, minlength=faixas)
    total = hist.sum()
    return {f"hog_{i}": float(hist[i] / total) if total > 0 else 0.0 for i in range(faixas)}


def descrever_textura(cinza: np.ndarray, lab: np.ndarray, codigos_lbp: np.ndarray,
                      mascara: np.ndarray) -> dict[str, float]:
    """Intensidade, cor, histograma, LBP, GLCM e HOG dos pixels da máscara (recorte da caixa)."""
    x, y, w, h = cv2.boundingRect(mascara)
    y0, y1, x0, x1 = max(y - 1, 0), y + h + 1, max(x - 1, 0), x + w + 1
    m, c = mascara[y0:y1, x0:x1], cinza[y0:y1, x0:x1]
    dentro = m > 0
    f = {"media_cinza": float(c[dentro].mean()), "desvio_cinza": float(c[dentro].std()),
         "media_a": float(lab[y0:y1, x0:x1, 1][dentro].mean()),
         "media_b": float(lab[y0:y1, x0:x1, 2][dentro].mean()),
         "desvio_a": float(lab[y0:y1, x0:x1, 1][dentro].std()),
         "desvio_b": float(lab[y0:y1, x0:x1, 2][dentro].std())}
    f.update(estatisticas_histograma(c[dentro]))
    lb = estatisticas_histograma(codigos_lbp[y0:y1, x0:x1][dentro])
    f["lbp_entropia"], f["lbp_uniformidade"] = lb["entropia"], lb["uniformidade"]
    f.update(glcm(c, m))
    f.update(hog(c, m))
    return f


def ficha(ctx, mascaras: list[np.ndarray], refs) -> dict[str, float]:
    """Atributos da região suspeita (união das máscaras dos especialistas), da diferença do "500"
    para o modelo e os globais."""
    f = {a: 0.0 for a in ATRIBUTOS if a not in ESTAGIOS}
    f.update(diferenca_do_500(ctx, refs))
    cinza = ctx.suave
    lab = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2LAB)
    codigos = lbp(cinza)

    # Global: a cápsula inteira, sem a faixa do contorno (extensão opcional)
    interior = cv2.erode(ctx.capsula, _ee(2 * 13 + 1))
    for a, v in descrever_textura(cinza, lab, codigos, interior).items():
        f["g_" + a] = v

    regiao = np.zeros_like(ctx.capsula)
    for m in mascaras:
        regiao |= m
    regiao &= ctx.capsula
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(regiao, connectivity=8)
    if n <= 1:
        return f

    # --- geometria
    x, y, w, h = cv2.boundingRect(regiao)
    maior = np.where(rotulos == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])), 255, 0).astype(np.uint8)
    forma = descrever_forma(maior)
    mo = cv2.moments(maior, binaryImage=True)
    mu20, mu02, mu11 = mo["mu20"] / mo["m00"], mo["mu02"] / mo["m00"], mo["mu11"] / mo["m00"]
    l1 = (mu20 + mu02) / 2 + np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    angulo = abs(np.degrees(0.5 * np.arctan2(2 * mu11, mu20 - mu02)))   # 0 = ao longo da cápsula
    f.update({"n_manchas": float(n - 1), "area_total": float(np.count_nonzero(regiao)),
              "largura_caixa": float(w), "altura_caixa": float(h),
              "eixo_maior": float(4 * np.sqrt(max(l1, 0))), "orientacao": float(angulo),
              **{k: forma[k] for k in ("area_maior", "perimetro", "compacidade",
                                       "excentricidade", "extensao", "solidez")}})

    # --- textura, cor
    f.update(descrever_textura(cinza, lab, codigos, regiao))
    anel = cv2.dilate(regiao, _ee(2 * LARGURA_ANEL + 1)) & ~regiao & ctx.capsula
    if anel.any():
        f["contraste_anel"] = float(cinza[regiao > 0].mean() - cinza[anel > 0].mean())

    # --- posição
    dentro = regiao > 0
    xs = np.where(ctx.capsula.any(0))[0]
    dist = cv2.distanceTransform(ctx.capsula, cv2.DIST_L2, 3)
    sobre_500 = cv2.dilate(ctx.impressao, _ee(DILATA_IMPRESSAO)) > 0
    f.update({"frac_tampa_preta": float(np.mean(np.where(dentro)[1] < ctx.x_divisa)),
              "frac_sobre_500": float(np.mean(sobre_500[dentro])),
              "dist_contorno": float(dist[dentro].mean()),
              "posicao_x": float((mo["m10"] / mo["m00"] - xs[0]) / max(xs[-1] - xs[0], 1))})
    return f


def _ficha_linha(l: dict) -> dict:
    ctx = preparar(carregar_imagem(l["caminho"]))
    mascaras = [cv2.imread(str(DIR_MASCARAS / f"{_chave(l['arquivo'])}__{e}.png"),
                           cv2.IMREAD_GRAYSCALE) for e in ESTAGIOS]
    return ficha(ctx, mascaras, cascata.referencias())


def calcular_atributos() -> None:
    linhas = carregar_particao("calibracao") + carregar_particao("classificacao")
    d = carregar_valores()
    assert list(d["arquivo"]) == [l["arquivo"] for l in linhas], "rode src/cascata.py --recalcular"
    print("Montando as referências com as 30 boas do banco...")
    refs = referencias_do_banco()
    print(f"Descrevendo a região suspeita de {len(linhas)} imagens...")
    fichas = mapear(_ficha_linha, linhas, refs)
    with open(CSV_ATRIBUTOS, "w", newline="", encoding="utf-8") as arq:
        w = csv.DictWriter(arq, fieldnames=["arquivo", "tipo", "rotulo", "conjunto", "fold"] + ATRIBUTOS,
                           lineterminator="\n")
        w.writeheader()
        for i, (l, f) in enumerate(zip(linhas, fichas)):
            f.update({e: float(d[e][i]) for e in ESTAGIOS})
            w.writerow({"arquivo": l["arquivo"], "tipo": l["tipo"], "rotulo": l["rotulo"],
                        "conjunto": l["conjunto"], "fold": "" if l["fold"] is None else l["fold"],
                        **{a: round(f[a], 5) for a in ATRIBUTOS}})


def carregar_atributos() -> dict:
    with open(CSV_ATRIBUTOS, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    d = {"arquivo": np.array([l["arquivo"] for l in linhas]),
         "tipo": np.array([l["tipo"] for l in linhas]),
         "rotulo": np.array([l["rotulo"] for l in linhas]),
         "conjunto": np.array([l["conjunto"] for l in linhas]),
         "fold": np.array([int(l["fold"]) if l["fold"] != "" else -1 for l in linhas])}
    for a in ATRIBUTOS:
        d[a] = np.array([float(l[a]) for l in linhas])
    return d


# --- 2. CLASSIFICADORES (mesmo protocolo da Etapa 5) ---
def modelos() -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    interno = StratifiedKFold(3, shuffle=True, random_state=SEMENTE)
    svm = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
        {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]},
        scoring="balanced_accuracy", cv=interno, n_jobs=-1)
    rf = GridSearchCV(
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEMENTE),
        {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]},
        scoring="balanced_accuracy", cv=interno, n_jobs=-1)
    return {"SVM": svm, "RandomForest": rf}


def validar(X: np.ndarray, y: np.ndarray, fold: np.ndarray, modelo, positiva: str | None) -> dict:
    """5 rodadas com os folds da partição; imagens de calibração (fold -1) só treinam."""
    pred = np.empty(len(y), dtype=object)
    pont = np.zeros(len(y))
    imp = []
    for k in range(5):
        tr, te = fold != k, fold == k
        modelo.fit(X[tr], y[tr])
        est = modelo.best_estimator_
        pred[te] = est.predict(X[te])
        if positiva is not None:
            pos = list(est.classes_).index(positiva)
            if hasattr(est, "predict_proba"):
                pont[te] = est.predict_proba(X[te])[:, pos]
            else:
                s = est.decision_function(X[te])
                pont[te] = s if pos == 1 else -s
        if hasattr(est, "feature_importances_"):
            imp.append(est.feature_importances_)
    return {"pred": pred, "pontuacao": pont, "importancia": np.mean(imp, 0) if imp else None}


def metricas_binaria(y: np.ndarray, pred: np.ndarray, pont: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import roc_auc_score
    pos, neg = y == "defeituosa", y == "normal"
    acusou = pred == "defeituosa"
    rec, esp = acusou[pos].mean(), 1 - acusou[neg].mean()
    prec = acusou[pos].sum() / max(acusou.sum(), 1)
    return {"acuracia_balanceada": (rec + esp) / 2, "precisao": prec, "recall": rec,
            "especificidade": esp, "f1_defeituosa": 2 * prec * rec / max(prec + rec, 1e-9),
            "roc_auc": roc_auc_score(pos, pont)}


def metricas_tipo(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score
    m = {"acuracia_balanceada": balanced_accuracy_score(y, pred),
         "f1_macro": f1_score(y, pred, average="macro", labels=TIPOS, zero_division=0)}
    for t in TIPOS:
        m[f"acerto_{t}"] = float(np.mean(pred[y == t] == t))
    return m


def tabela(linhas: list[dict], nome: str, cabecalho: list[str]) -> str:
    chaves = list(linhas[0].keys())
    with open(DIR_RESULT / f"{nome}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=chaves, lineterminator="\n")
        w.writeheader()
        for l in linhas:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in l.items()})
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    larg = [max([len(k)] + [len(fmt(l[k])) for l in linhas]) for k in chaves]
    txt = cabecalho + ["", "  ".join(k.ljust(w) for k, w in zip(chaves, larg))]
    txt += ["  ".join(fmt(l[k]).ljust(w) for k, w in zip(chaves, larg)) for l in linhas]
    texto = "\n".join(txt)
    (DIR_RESULT / f"{nome}.txt").write_text(texto + "\n", encoding="utf-8")
    return texto


# --- 3. FIGURAS ---
def figura_roc(y: np.ndarray, curvas: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fig, ax = plt.subplots(figsize=(6.6, 6), facecolor=FUNDO)
    cascata._eixos(ax)
    cores = {"cascata (maior valor / limiar)": TINTA, "Random Forest": AZUL, "SVM": LARANJA}
    for nome, s in curvas.items():
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(100 * fpr, 100 * tpr, color=cores[nome], linewidth=2,
                label=f"{nome}: AUC {roc_auc_score(y, s):.3f}")
    ax.plot([0, 100], [0, 100], color=GRADE, linewidth=1, linestyle="--")
    ax.set_xlim(-1, 60)
    ax.set_ylim(40, 101)
    ax.set_xlabel("boas rejeitadas - alarme falso (%)", color=TINTA_2)
    ax.set_ylabel("defeituosas encontradas - recall (%)", color=TINTA_2)
    ax.set_title("Normal x defeituosa: curvas ROC (ficha completa, 5 folds)", color=TINTA,
                 loc="left", fontsize=12)
    ax.legend(frameon=False, loc="lower right", labelcolor=TINTA)
    fig.tight_layout()
    fig.savefig(DIR_RESULT / "roc_binaria.png", dpi=110, facecolor=FUNDO)
    plt.close(fig)


def figura_confusao(y: np.ndarray, previsoes: dict[str, np.ndarray]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from sklearn.metrics import confusion_matrix

    colunas = TIPOS + ["good"]
    nomes = [NOMES[t] for t in TIPOS] + ["não achada"]
    mapa = LinearSegmentedColormap.from_list("azul", [FUNDO, AZUL])
    fig, eixos = plt.subplots(1, len(previsoes), figsize=(7.2 * len(previsoes), 5.4), facecolor=FUNDO)
    for ax, (titulo, pred) in zip(np.atleast_1d(eixos), previsoes.items()):
        cm = confusion_matrix(y, pred, labels=colunas)[:len(TIPOS)]
        ax.imshow(cm, cmap=mapa, vmin=0, vmax=23, aspect="auto")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=11,
                        color="white" if cm[i, j] > 13 else TINTA)
        ax.set_xticks(range(len(colunas)))
        ax.set_xticklabels(nomes, rotation=25, ha="right", color=TINTA)
        ax.set_yticks(range(len(TIPOS)))
        ax.set_yticklabels(nomes[:len(TIPOS)], color=TINTA)
        ax.set_xlabel("tipo apontado", color=TINTA_2)
        ax.set_ylabel("tipo verdadeiro", color=TINTA_2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(titulo, color=TINTA, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(DIR_RESULT / "matriz_confusao_tipo.png", dpi=100, facecolor=FUNDO)
    plt.close(fig)


def figura_importancia(nomes: list[str], imp: np.ndarray, n: int = 25) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    cores = dict(zip(GRUPOS, [TINTA, AZUL, LARANJA, "#1baf7a", "#8d5fd3"]))
    cor = {a: cores[g] for g, cols in GRUPOS.items() for a in cols}
    ordem = np.argsort(imp)[-n:]
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=FUNDO)
    cascata._eixos(ax)
    ax.grid(axis="y", visible=False)
    rotulo = lambda a: NOMES[a] + " (valor)" if a in ESTAGIOS else a
    ax.barh([rotulo(nomes[i]) for i in ordem], imp[ordem], color=[cor[nomes[i]] for i in ordem])
    ax.legend(handles=[Patch(color=c, label=g) for g, c in cores.items()], loc="lower right",
              frameon=False, labelcolor=TINTA)
    ax.set_xlabel("importância média (Random Forest, 5 rodadas)", color=TINTA_2)
    ax.set_title(f"Qual defeito: os {n} atributos que o Random Forest mais usa", color=TINTA,
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(DIR_RESULT / "importancia_atributos_cascata.png", dpi=100, facecolor=FUNDO)
    plt.close(fig)


# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 7: classificadores sobre os especialistas")
    parser.add_argument("--recalcular", action="store_true",
                        help="refaz a ficha de atributos (dados/atributos_cascata.csv)")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    if args.recalcular or not CSV_ATRIBUTOS.exists():
        calcular_atributos()
    d = carregar_atributos()
    dv = carregar_valores()
    assert list(d["arquivo"]) == list(dv["arquivo"]), "atributos desatualizados: use --recalcular"
    cls = d["conjunto"] == "classificacao"
    ruins = cls & (d["rotulo"] == "defeituosa")
    r_cascata = decidir(dv, ALFA)
    X = lambda cols: np.column_stack([d[c] for c in cols])

    conjuntos = {**GRUPOS, "completo": COMPLETO}
    ablacao = {"completo": COMPLETO,
               "completo sem cor": [a for a in COMPLETO if a not in COR],
               "completo sem orientacao": [a for a in COMPLETO if a != "orientacao"],
               "completo sem GLCM": [a for a in COMPLETO if a not in GLCM],
               "completo sem HOG": [a for a in COMPLETO if a not in HOG],
               "completo sem os 4": [a for a in COMPLETO if a not in COR + GLCM + HOG + ["orientacao"]],
               "completo sem diferenca": [a for a in COMPLETO if a not in GRUPOS["diferenca"]],
               "valores + textura da regiao": GRUPOS["valores"] + GRUPOS["textura"],
               "valores + textura global": GRUPOS["valores"] + GLOBAL,
               "textura global": GLOBAL}
    res = {}

    # ---------------- binária
    y = d["rotulo"]
    pont_cascata = r_cascata["razao"].max(1)
    pred_cascata = np.where(r_cascata["defeituosa"], "defeituosa", "normal")
    tab = [{"atributos": "valores", "decisao": "cascata (limiares)",
            **metricas_binaria(y[cls], pred_cascata[cls], pont_cascata[cls])}]
    for nome, cols in {**conjuntos, **ablacao}.items():
        for clf, modelo in modelos().items():
            r = validar(X(cols), y, d["fold"], modelo, "defeituosa")
            res[("binaria", nome, clf)] = r
            m = metricas_binaria(y[cls], r["pred"][cls], r["pontuacao"][cls])
            if nome in conjuntos:
                tab.append({"atributos": nome, "decisao": clf, **m})
            print(f"  binaria  {nome:28s} {clf:13s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    print("\n" + tabela(tab, "classificadores_binaria", [
        "Etapa 7 - normal x defeituosa (301 imagens, 5 folds): regra da cascata x classificadores",
        f"cascata = um limiar por etapa (ALFA = {ALFA}); classificadores na decisão padrão",
        "atributos: valores = os 5 valores dos especialistas; geometria, textura e posicao = da "
        "região suspeita (união das máscaras); diferenca = do \"500\" para o modelo casado"]) + "\n")

    # ---------------- tipo (109 defeituosas)
    yt, ft = d["tipo"][ruins], d["fold"][ruins]
    tipo_simples = cascata.tipo_apontado(r_cascata, refinar=False)
    tipo_cascata = cascata.tipo_apontado(r_cascata)
    tab_t = [{"atributos": "valores", "decisao": "cascata: 1ª etapa que acusa",
              **metricas_tipo(yt, tipo_simples[ruins])},
             {"atributos": "valores + diferenca", "decisao": "cascata: 1ª etapa + diferença do 500",
              **metricas_tipo(yt, tipo_cascata[ruins])}]
    for nome, cols in {**conjuntos, **ablacao}.items():
        for clf, modelo in modelos().items():
            r = validar(X(cols)[ruins], yt, ft, modelo, None)
            res[("tipo", nome, clf)] = r
            m = metricas_tipo(yt, r["pred"])
            if nome in conjuntos:
                tab_t.append({"atributos": nome, "decisao": clf, **m})
            print(f"  tipo     {nome:28s} {clf:13s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    print("\n" + tabela(tab_t, "classificadores_tipo", [
        "Etapa 7 - qual defeito (109 defeituosas, 5 classes, 5 folds); acaso = 0,20",
        "nas regras da cascata, defeituosa que nenhuma etapa acusa conta como erro; os "
        "classificadores dão um tipo a todas as 109 (ver sistema_final.txt para a comparação "
        "nas mesmas condições)"]) + "\n")

    # ---------------- ablação dos atributos novos e região x global
    txt = ["Etapa 7 - o que cada atributo do pipeline sugerido acrescenta (acurácia balanceada)",
           "cor = a*, b* do Lab; orientacao = ângulo do eixo maior; GLCM = 4 descritores de "
           "Haralick; HOG = 8 orientações do gradiente; diferenca = do \"500\" para o modelo",
           "(uma semente do Random Forest varia +-0,01; com ~20 cápsulas por tipo, diferenças "
           "menores que ~0,05 não são conclusivas)",
           "região = pixels da região suspeita; global = cápsula inteira (extensão opcional)", "",
           f"{'atributos':<30}{'n':>4}{'binária SVM':>13}{'binária RF':>12}{'tipo SVM':>10}{'tipo RF':>9}"]
    for nome, cols in ablacao.items():
        b = [metricas_binaria(y[cls], res[("binaria", nome, c)]["pred"][cls],
                              res[("binaria", nome, c)]["pontuacao"][cls])["acuracia_balanceada"]
             for c in ("SVM", "RandomForest")]
        t = [metricas_tipo(yt, res[("tipo", nome, c)]["pred"])["acuracia_balanceada"]
             for c in ("SVM", "RandomForest")]
        txt.append(f"{nome:<30}{len(cols):4d}{b[0]:13.3f}{b[1]:12.3f}{t[0]:10.3f}{t[1]:9.3f}")
    texto = "\n".join(txt)
    (DIR_RESULT / "ablacao_atributos.txt").write_text(texto + "\n", encoding="utf-8")
    print(texto + "\n")

    # ---------------- sistema final: cascata decide, classificador dá o tipo
    grupo, clf = CONFIG_SISTEMA
    pred_tipo = np.full(len(y), "good", dtype=object)
    pred_tipo[ruins] = res[("tipo", grupo, clf)]["pred"]
    # boas acusadas pela cascata também recebem um tipo: modelo de cada rodada treinado nas
    # defeituosas de treino (não muda a nota do tipo, que só olha as defeituosas)
    modelo = modelos()[clf]
    boas_acusadas = cls & (d["rotulo"] == "normal") & r_cascata["defeituosa"]
    for k in range(5):
        te = boas_acusadas & (d["fold"] == k)
        if te.any():
            tr = ruins & (d["fold"] != k)
            modelo.fit(X(conjuntos[grupo])[tr], d["tipo"][tr])
            pred_tipo[te] = modelo.predict(X(conjuntos[grupo])[te])
    sistema = np.where(r_cascata["defeituosa"], pred_tipo, "good").astype(object)
    colunas = ["good"] + TIPOS
    txt = ["Etapa 7 - sistema final: a cascata decide normal x defeituosa "
           f"(ALFA = {ALFA}) e o classificador dá o tipo",
           f"classificador do tipo: {clf}, ficha '{grupo}' ({len(conjuntos[grupo])} atributos), "
           "treinado só nas defeituosas de treino de cada rodada", ""]
    for nome, ap in (("regra simples: 1ª etapa que acusa", tipo_simples),
                     ("regra adotada: 1ª etapa + diferença do \"500\"", tipo_cascata),
                     (f"cascata + {clf}", sistema)):
        achadas = ruins & r_cascata["defeituosa"]
        ac = [float(np.mean(ap[cls & (d["tipo"] == t)] == t)) for t in TIPOS]
        f1 = cascata.f1_macro_tipos(d["tipo"][cls], ap[cls])
        txt += [f"{nome}", f"{'verdadeiro':<18}" + "".join(f"{t:>16}" for t in colunas) + "   acerto"]
        for t in colunas:
            sel = cls & (d["tipo"] == t)
            txt.append(f"{t:<18}" + "".join(f"{int((ap[sel] == p).sum()):>16d}" for p in colunas)
                       + f"{np.mean(ap[sel] == t):9.2f}")
        txt += [f"tipo apontado (acurácia balanceada nas 5 classes, não achada = erro): {np.mean(ac):.3f}",
                f"F1 macro dos 5 tipos (boas acusadas contam como falso positivo do tipo): {f1:.3f}",
                f"entre as defeituosas achadas, tipo certo em {np.mean(ap[achadas] == d['tipo'][achadas]):.0%}", ""]
    texto = "\n".join(txt)
    (DIR_RESULT / "sistema_final.txt").write_text(texto, encoding="utf-8")
    print(texto)

    # ---------------- figuras
    figura_roc((y == "defeituosa")[cls], {
        "cascata (maior valor / limiar)": pont_cascata[cls],
        "Random Forest": res[("binaria", "completo", "RandomForest")]["pontuacao"][cls],
        "SVM": res[("binaria", "completo", "SVM")]["pontuacao"][cls]})
    figura_confusao(d["tipo"][ruins], {
        "Regra simples: primeira etapa que acusa": tipo_simples[ruins],
        "Regra adotada: + diferença do \"500\"": tipo_cascata[ruins],
        f"Cascata + {clf} (ficha {grupo})": sistema[ruins]})
    figura_importancia(COMPLETO, res[("tipo", "completo", "RandomForest")]["importancia"])
    print(f"Resultados em {DIR_RESULT}")


if __name__ == "__main__":
    main()
