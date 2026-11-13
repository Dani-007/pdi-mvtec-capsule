"""
Etapa 5 - Classificação a partir das fichas de atributos (dados/atributos.csv).

Duas tarefas, em sequência (como numa linha de inspeção):
    binaria   normal x defeituosa, nas 301 imagens do conjunto 'classificacao' (coluna 'rotulo')
    defeitos  qual defeito: crack, faulty_imprint, poke, scratch, squeeze, só nas 109
              defeituosas (coluna 'tipo', 5 classes)

Comparações pedidas no enunciado:
    - grupos de atributos: forma, textura, posicao_impressao e todos juntos;
    - classificadores: SVM (núcleo RBF) e Random Forest.

Protocolo (sem vazamento):
    - validação cruzada com a coluna 'fold' da partição (5 rodadas: treina em 4 folds e testa
      no 5º), então cada imagem é testada uma vez, por um modelo que não a viu;
    - padronização (z-score) e escolha de hiperparâmetros feitas SÓ no treino de cada rodada
      (busca em grade com validação cruzada interna de 3 partes);
    - classes desbalanceadas: class_weight='balanced'.

Ponto de operação da tarefa binária. O objetivo do projeto é ACHAR os defeitos (rejeitar uma
boa é um erro menor que aprovar uma defeituosa). Por isso, além da decisão padrão do
classificador, a binária é avaliada com um limiar que prioriza os defeitos: em cada rodada, o
limiar sobre a pontuação "defeituosa" é escolhido SÓ no treino (previsões fora da amostra da
validação interna) como o maior valor que ainda dá recall >= RECALL_ALVO nas defeituosas do
treino. Depois ele é aplicado, sem mudança, às imagens de teste da rodada.

Métricas: acurácia balanceada, precisão, recall e especificidade, F1 (binária: classe
defeituosa; defeitos: média macro), ROC-AUC (binária), matriz de confusão e acerto por tipo.
Referência do acaso: acurácia balanceada 0,50 (binária) e 0,20 (defeitos, 5 classes).

Saídas em resultados/etapa5/:
    tabela_binaria.csv|txt                            binária, decisão padrão
    tabela_binaria_prioriza_defeitos.csv|txt          binária, limiar para recall >= RECALL_ALVO
    encontradas_por_tipo.txt                          binária (prioriza defeitos), por tipo
    tabela_defeitos.csv|txt                           qual defeito
    acerto_por_tipo.txt                               acerto por tipo (tarefa defeitos)
    matriz_confusao_*.png                             melhor configuração de cada tarefa
    importancia_atributos.png                         Random Forest, todos os atributos
    erros_binaria.csv, exemplos_erros.png             erros da binária (prioriza defeitos)

Uso:  python src/classificacao.py      (exige dados/atributos.csv; ~2 min)
"""

from __future__ import annotations

import csv

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from atributos import GRUPOS, SAIDA as CSV_ATRIBUTOS
from pipeline import RAIZ

DIR_RESULT = RAIZ / "resultados" / "etapa5"
SEMENTE = 42
RECALL_ALVO = 0.90      # tarefa binária: fração mínima de defeituosas a encontrar (no treino)
DEFEITOS = ["crack", "faulty_imprint", "poke", "scratch", "squeeze"]
GRUPOS_CMP = {**GRUPOS, "todos": [a for g in GRUPOS.values() for a in g]}


def modelos() -> dict[str, GridSearchCV]:
    interno = StratifiedKFold(3, shuffle=True, random_state=SEMENTE)
    svm = GridSearchCV(
        make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced")),
        {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]},
        scoring="balanced_accuracy", cv=interno)
    rf = GridSearchCV(
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEMENTE),
        {"max_depth": [None, 5, 10], "min_samples_leaf": [1, 3]},
        scoring="balanced_accuracy", cv=interno)
    return {"SVM": svm, "RandomForest": rf}


def carregar() -> list[dict]:
    with open(CSV_ATRIBUTOS, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _pontuar(est, X: np.ndarray) -> np.ndarray:
    """Pontuação de "quão defeituosa": probabilidade (Random Forest) ou distância à fronteira
    de decisão (SVM)."""
    pos = list(est.classes_).index("defeituosa")
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, pos]
    d = est.decision_function(X)            # SVM binário: > 0 favorece classes_[1]
    return d if pos == 1 else -d


def limiar_para_recall(est, X: np.ndarray, y: np.ndarray) -> float:
    """Limiar escolhido SÓ com o treino: pontuações fora da amostra (validação interna de 3
    partes) e o maior limiar que ainda encontra >= RECALL_ALVO das defeituosas do treino."""
    interno = StratifiedKFold(3, shuffle=True, random_state=SEMENTE)
    metodo = "predict_proba" if hasattr(est, "predict_proba") else "decision_function"
    s = cross_val_predict(clone(est), X, y, cv=interno, method=metodo)
    classes = sorted(set(y))
    if metodo == "predict_proba":
        s = s[:, classes.index("defeituosa")]
    elif classes.index("defeituosa") == 0:
        s = -s
    defeituosas = np.sort(s[y == "defeituosa"])
    k = int(np.floor((1 - RECALL_ALVO) * len(defeituosas)))
    return float(defeituosas[k])


def validacao_cruzada(X: np.ndarray, y: np.ndarray, folds: np.ndarray, modelo) -> dict:
    """Roda as 5 rodadas e devolve, para cada imagem, a classe prevista (decisão padrão) e, na
    tarefa binária, a pontuação de "defeituosa" e a decisão com o limiar que prioriza defeitos."""
    pred = np.empty(len(y), dtype=object)
    pred_rec = np.empty(len(y), dtype=object)
    pontuacao = np.zeros(len(y))
    importancias = []
    binaria = "defeituosa" in set(y)
    for k in range(5):
        tr, te = folds != k, folds == k
        modelo.fit(X[tr], y[tr])
        est = modelo.best_estimator_
        pred[te] = modelo.predict(X[te])
        if binaria:
            pontuacao[te] = _pontuar(est, X[te])
            t = limiar_para_recall(est, X[tr], y[tr])
            pred_rec[te] = np.where(pontuacao[te] >= t, "defeituosa", "normal")
        if hasattr(est, "feature_importances_"):
            importancias.append(est.feature_importances_)
    return {"pred": pred, "pred_recall": pred_rec, "pontuacao": pontuacao,
            "importancia": np.mean(importancias, axis=0) if importancias else None}


def metricas(tarefa: str, y: np.ndarray, r: dict, chave: str = "pred") -> dict[str, float]:
    pred = r[chave]
    m = {"acuracia_balanceada": balanced_accuracy_score(y, pred)}
    if tarefa == "binaria":
        m["precisao"] = precision_score(y, pred, pos_label="defeituosa", zero_division=0)
        m["recall"] = recall_score(y, pred, pos_label="defeituosa")
        m["especificidade"] = recall_score(y, pred, pos_label="normal")
        m["f1_defeituosa"] = f1_score(y, pred, pos_label="defeituosa")
        m["roc_auc"] = roc_auc_score(y == "defeituosa", r["pontuacao"])
    else:
        m["f1_macro"] = f1_score(y, pred, average="macro")
    return m


def salvar_tabela(nome: str, linhas: list[dict]) -> str:
    chaves = list(linhas[0].keys())
    with open(DIR_RESULT / f"{nome}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=chaves, lineterminator="\n")
        w.writeheader()
        for l in linhas:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in l.items()})
    larg = [max([len(k)] + [len(f"{l[k]:.3f}" if isinstance(l[k], float) else str(l[k]))
                            for l in linhas]) for k in chaves]
    txt = ["  ".join(k.ljust(w) for k, w in zip(chaves, larg))]
    for l in linhas:
        txt.append("  ".join((f"{l[k]:.3f}" if isinstance(l[k], float) else str(l[k])).ljust(w)
                             for k, w in zip(chaves, larg)))
    texto = "\n".join(txt)
    (DIR_RESULT / f"{nome}.txt").write_text(texto + "\n", encoding="utf-8")
    return texto


def figura_confusao(nome: str, titulo: str, y: np.ndarray, pred: np.ndarray, rotulos: list[str]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y, pred, labels=rotulos)
    fig, ax = plt.subplots(figsize=(1.3 * len(rotulos) + 2.5, 1.1 * len(rotulos) + 2))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(rotulos)):
        for j in range(len(rotulos)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(rotulos)))
    ax.set_xticklabels(rotulos, rotation=30, ha="right")
    ax.set_yticks(range(len(rotulos)))
    ax.set_yticklabels(rotulos)
    ax.set_xlabel("previsto")
    ax.set_ylabel("verdadeiro")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(DIR_RESULT / nome, dpi=90)
    plt.close(fig)


def figura_importancia(nomes: list[str], imp: np.ndarray):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cor = {a: c for g, c in zip(GRUPOS, ["#2a78d6", "#1baf7a", "#e0702a"]) for a in GRUPOS[g]}
    ordem = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(8, 9))
    ax.barh([nomes[i] for i in ordem], imp[ordem], color=[cor[nomes[i]] for i in ordem])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=g)
                       for g, c in zip(GRUPOS, ["#2a78d6", "#1baf7a", "#e0702a"])],
              loc="lower right", frameon=False)
    ax.set_xlabel("importância média (Random Forest, 5 rodadas)")
    ax.set_title("Quais atributos o Random Forest mais usa (tarefa: qual defeito)")
    fig.tight_layout()
    fig.savefig(DIR_RESULT / "importancia_atributos.png", dpi=90)
    plt.close(fig)


def figura_erros(linhas: list[dict], pred: np.ndarray, prob_def: np.ndarray):
    """prob_def: pontuação de 'defeituosa' do melhor modelo (probabilidade no Random Forest)."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pipeline import DATASET_DIR
    from segmentacao import DIR_MASCARAS

    rot = np.array([l["rotulo"] for l in linhas])
    fp = [i for i in np.argsort(-prob_def) if rot[i] == "normal" and pred[i] == "defeituosa"][:3]
    fn = [i for i in np.argsort(prob_def) if rot[i] == "defeituosa" and pred[i] == "normal"][:3]
    with open(DIR_RESULT / "erros_binaria.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["arquivo", "tipo", "erro", "pontuacao_defeituosa"])
        for i in range(len(linhas)):
            if pred[i] != rot[i]:
                w.writerow([linhas[i]["arquivo"], linhas[i]["tipo"],
                            "falso positivo" if rot[i] == "normal" else "falso negativo",
                            round(float(prob_def[i]), 3)])
    casos = [(i, "FALSO POSITIVO") for i in fp] + [(i, "FALSO NEGATIVO") for i in fn]
    if not casos:
        return
    fig, eixos = plt.subplots(2, 3, figsize=(16, 7))
    for ax in eixos.ravel():
        ax.axis("off")
    for k, (i, tipo_erro) in enumerate(casos):
        ax = eixos[0 if tipo_erro.startswith("FALSO P") else 1, k % 3]
        arq = linhas[i]["arquivo"]
        img = cv2.cvtColor(cv2.imread(str(DATASET_DIR / arq)), cv2.COLOR_BGR2RGB)
        m = cv2.imread(str(DIR_MASCARAS / arq.replace("/", "_")), cv2.IMREAD_GRAYSCALE)
        if m is not None and m.any():
            img[m > 0] = (0.45 * img[m > 0] + 0.55 * np.array([235, 104, 52])).astype(np.uint8)
        ax.imshow(img[150:850, :])
        ax.set_title(f"{tipo_erro}\n{arq} (pontuação={prob_def[i]:.2f})", fontsize=10)
    fig.suptitle("Etapa 5 - erros da binária priorizando defeitos (laranja = mancha da Etapa 3)")
    fig.tight_layout()
    fig.savefig(DIR_RESULT / "exemplos_erros.png", dpi=80)
    plt.close(fig)


def main() -> None:
    DIR_RESULT.mkdir(parents=True, exist_ok=True)
    todas = carregar()
    tarefas = {"binaria": (todas, "rotulo"),
               "defeitos": ([l for l in todas if l["rotulo"] == "defeituosa"], "tipo")}

    resultados = {}
    for tarefa, (linhas, coluna) in tarefas.items():
        y = np.array([l[coluna] for l in linhas])
        folds = np.array([int(l["fold"]) for l in linhas])
        tab, tab_rec = [], []
        for grupo, cols in GRUPOS_CMP.items():
            X = np.array([[float(l[c]) for c in cols] for l in linhas])
            for nome, modelo in modelos().items():
                r = validacao_cruzada(X, y, folds, modelo)
                m = metricas(tarefa, y, r)
                resultados[(tarefa, grupo, nome)] = r
                tab.append({"grupo": grupo, "classificador": nome, **m})
                if tarefa == "binaria":
                    tab_rec.append({"grupo": grupo, "classificador": nome,
                                    **metricas(tarefa, y, r, "pred_recall")})
                print(f"  {tarefa:8s} {grupo:18s} {nome:13s} " +
                      "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
        print(f"\n== {tarefa} ==\n" + salvar_tabela(f"tabela_{tarefa}", tab) + "\n")
        if tab_rec:
            print(f"== {tarefa}, prioriza defeitos (recall alvo {RECALL_ALVO:.0%} no treino) ==\n"
                  + salvar_tabela("tabela_binaria_prioriza_defeitos", tab_rec) + "\n")

        melhor = max(tab, key=lambda l: l["acuracia_balanceada"])
        r = resultados[(tarefa, melhor["grupo"], melhor["classificador"])]
        rotulos = ["normal", "defeituosa"] if tarefa == "binaria" else DEFEITOS
        figura_confusao(f"matriz_confusao_{tarefa}.png",
                        f"{tarefa}: {melhor['grupo']} + {melhor['classificador']} "
                        f"(acur. bal. {melhor['acuracia_balanceada']:.2f})", y, r["pred"], rotulos)
        if tarefa == "defeitos":
            acerto = {t: float(np.mean(r["pred"][y == t] == t)) for t in DEFEITOS}
            txt = "Acerto por tipo (melhor configuração):\n" + \
                  "\n".join(f"  {t:15s} {v:.2f}" for t, v in acerto.items())
            (DIR_RESULT / "acerto_por_tipo.txt").write_text(txt + "\n", encoding="utf-8")
            print(txt)
        if tarefa == "binaria":
            # Ponto de operação adotado: prioriza defeitos (melhor acurácia balanceada nele).
            mr = max(tab_rec, key=lambda l: l["acuracia_balanceada"])
            rr = resultados[(tarefa, mr["grupo"], mr["classificador"])]
            figura_confusao("matriz_confusao_binaria_prioriza_defeitos.png",
                            f"prioriza defeitos: {mr['grupo']} + {mr['classificador']}\n"
                            f"(recall {mr['recall']:.2f}, especificidade {mr['especificidade']:.2f})",
                            y, rr["pred_recall"], rotulos)
            tipos = np.array([l["tipo"] for l in linhas])
            txt = "Defeituosas encontradas por tipo (binária, prioriza defeitos):\n" + "\n".join(
                f"  {t:15s} {np.sum(rr['pred_recall'][tipos == t] == 'defeituosa'):3d} / "
                f"{np.sum(tipos == t)}" for t in DEFEITOS)
            (DIR_RESULT / "encontradas_por_tipo.txt").write_text(txt + "\n", encoding="utf-8")
            print(txt)
            figura_erros(linhas, rr["pred_recall"], rr["pontuacao"])

    imp = resultados[("defeitos", "todos", "RandomForest")]["importancia"]
    figura_importancia(GRUPOS_CMP["todos"], imp)
    print(f"\nResultados em {DIR_RESULT}")


if __name__ == "__main__":
    main()
