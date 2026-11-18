"""
Etapa 7 - Cascata de especialistas: cada cápsula passa por 5 etapas, uma por defeito
(especialistas.py). Se alguma etapa acusar, a cápsula é defeituosa; se passar por todas,
é boa.

    imagem -> 1. amassado -> 2. impressão falha -> 3. rachadura -> 4. furinho -> 5. arranhão -> BOA
                   |               |                   |              |              |
                   +---------------+------- acusou: DEFEITUOSA -------+--------------+

Cada especialista devolve um valor; a etapa acusa quando valor > limiar da etapa.

Limiar de cada etapa (definido SÓ com imagens de treino):
    a = percentil (1 - ALFA) do valor nas cápsulas BOAS de treino, ou seja, a etapa aceita
        dar alarme falso em no máximo ALFA das boas;
    limiar = ponto médio entre `a` e o menor valor, acima de `a`, entre os defeitos-alvo de
        treino. Quando sobra folga entre boas e defeitos (amassado, impressão), o limiar vai
        para o meio da folga e a etapa praticamente não dá alarme falso.

Protocolo (o mesmo das Etapas 2-5, "quem ajusta não dá nota"):
    - referências dos especialistas: as 30 boas do conjunto 'banco';
    - limiares: validação cruzada com a coluna 'fold' (5 rodadas). Em cada rodada, o treino são
      as 20 boas de 'calibracao' + os 4 folds de treino; a decisão é tomada no 5º fold;
    - cada imagem de 'classificacao' é decidida uma vez, por limiares que não a viram.

Saídas:
    dados/especialistas.csv                     valor de cada especialista por imagem
    dados/limiares_cascata.json                 limiares finais (todas as imagens), para uso
    dados/gerado/especialistas/*.png            máscaras de cada especialista (não versionado)
    resultados/etapa7/tabela_especialistas.txt  AUC de cada especialista x tipo de defeito
    resultados/etapa7/tabela_cascata.txt|csv    cascata em vários pontos de operação (ALFA)
    resultados/etapa7/disparos_por_tipo.txt     que etapa acusa cada tipo; tipo apontado
    resultados/etapa7/ablacao.txt               cascata sem cada etapa
    resultados/etapa7/localizacao.txt           máscara da cascata x ground truth
    resultados/etapa7/erros.csv                 falsos positivos e falsos negativos
    resultados/etapa7/*.png                     figuras

Uso:
    python src/cascata.py                 (~1 min na 1ª vez; depois usa dados/especialistas.csv)
    python src/cascata.py --recalcular    refaz os valores dos especialistas
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
from pathlib import Path

import cv2
import numpy as np

from especialistas import (
    ESTAGIOS, NOMES, examinar, figura_passo_a_passo, referencias_do_banco,
)
from particao import carregar_particao
from pipeline import DATASET_DIR, RAIZ, carregar_imagem, carregar_mascara_gt, metricas_pixel

CSV_VALORES = RAIZ / "dados" / "especialistas.csv"
JSON_LIMIARES = RAIZ / "dados" / "limiares_cascata.json"
DIR_MASCARAS = RAIZ / "dados" / "gerado" / "especialistas"
DIR_RESULT = RAIZ / "resultados" / "etapa7"

TIPOS = ["crack", "faulty_imprint", "poke", "scratch", "squeeze"]
ALFAS = [0.01, 0.02, 0.03, 0.05, 0.08]   # pontos de operação avaliados
ALFA = 0.02              # ponto de operação padrão: cada etapa aceita ~2% de alarme falso
ALFA_PRIORIZA = 0.05     # ponto de operação que prioriza achar defeitos
MIN_PX_HIT = 30          # mesmo critério de "defeito localizado" das Etapas 1 e 3

AZUL, LARANJA = "#2a78d6", "#eb6834"
TINTA, TINTA_2, FUNDO, GRADE = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"


# =============================================================================
# 1. VALORES DOS ESPECIALISTAS (em paralelo)
# =============================================================================
_REFS = None   # referências do banco; os processos filhos herdam (fork)


def mapear(funcao, linhas: list[dict], refs) -> list:
    """Aplica funcao(linha) a cada imagem, em paralelo quando o sistema permite (fork)."""
    global _REFS
    _REFS = refs
    if "fork" not in mp.get_all_start_methods():
        return [funcao(l) for l in linhas]
    with mp.get_context("fork").Pool(min(8, os.cpu_count() or 1)) as pool:
        return pool.map(funcao, linhas, chunksize=4)


def referencias():
    return _REFS


def _chave(arquivo: str) -> str:
    return arquivo[:-4].replace("/", "_")


def _examinar_linha(l: dict) -> dict:
    valores, mascaras = examinar(carregar_imagem(l["caminho"]), _REFS)
    for nome, m in mascaras.items():
        cv2.imwrite(str(DIR_MASCARAS / f"{_chave(l['arquivo'])}__{nome}.png"), m)
    return valores


def calcular_valores() -> list[dict]:
    linhas = carregar_particao("calibracao") + carregar_particao("classificacao")
    print("Montando as referências com as 30 boas do banco...")
    refs = referencias_do_banco()
    DIR_MASCARAS.mkdir(parents=True, exist_ok=True)
    print(f"Passando {len(linhas)} imagens pelos 5 especialistas...")
    valores = mapear(_examinar_linha, linhas, refs)
    saida = [{"arquivo": l["arquivo"], "tipo": l["tipo"], "rotulo": l["rotulo"],
              "conjunto": l["conjunto"], "fold": "" if l["fold"] is None else l["fold"],
              **{e: round(v[e], 5) for e in ESTAGIOS}} for l, v in zip(linhas, valores)]
    with open(CSV_VALORES, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(saida[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(saida)
    return saida


def carregar_valores() -> dict[str, np.ndarray]:
    with open(CSV_VALORES, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    d = {"arquivo": np.array([l["arquivo"] for l in linhas]),
         "tipo": np.array([l["tipo"] for l in linhas]),
         "conjunto": np.array([l["conjunto"] for l in linhas]),
         "fold": np.array([int(l["fold"]) if l["fold"] != "" else -1 for l in linhas])}
    for e in ESTAGIOS:
        d[e] = np.array([float(l[e]) for l in linhas])
    return d


# =============================================================================
# 2. LIMIARES E DECISÃO
# =============================================================================
def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """ROC-AUC: chance de um positivo ter valor maior que um negativo (empate vale 1/2)."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    maior = (pos[:, None] > neg[None, :]).sum()
    igual = (pos[:, None] == neg[None, :]).sum()
    return float((maior + 0.5 * igual) / (len(pos) * len(neg)))


def limiar_etapa(boas: np.ndarray, alvo: np.ndarray, alfa: float) -> float:
    """Percentil (1 - alfa) das boas, levado ao meio da folga até o primeiro defeito-alvo."""
    a = float(np.percentile(boas, 100 * (1 - alfa)))
    acima = alvo[alvo > a]
    return (a + float(acima.min())) / 2 if len(acima) else a


def limiares(d: dict, treino: np.ndarray, alfa: float, margem: bool = True) -> dict[str, float]:
    boas = treino & (d["tipo"] == "good")
    saida = {}
    for e in ESTAGIOS:
        alvo = d[e][treino & (d["tipo"] == e)] if margem else np.array([])
        saida[e] = limiar_etapa(d[e][boas], alvo, alfa)
    return saida


def decidir(d: dict, alfa: float, estagios: list[str] = ESTAGIOS, margem: bool = True) -> dict:
    """Validação cruzada: limiares no treino de cada rodada, decisão no fold de teste.

    Devolve, para as imagens de 'classificacao': disparo[n, etapa] (bool), razao[n, etapa]
    (valor / limiar) e defeituosa[n]."""
    n = len(d["tipo"])
    cls = d["conjunto"] == "classificacao"
    razao = np.zeros((n, len(estagios)))
    for k in range(5):
        treino = (d["conjunto"] == "calibracao") | (cls & (d["fold"] != k))
        teste = cls & (d["fold"] == k)
        lim = limiares(d, treino, alfa, margem)
        for j, e in enumerate(estagios):
            razao[teste, j] = d[e][teste] / max(lim[e], 1e-9)
    disparo = razao > 1
    return {"disparo": disparo, "razao": razao, "defeituosa": disparo.any(1), "cls": cls,
            "estagios": estagios}


def resumo(d: dict, r: dict) -> dict:
    cls, det = r["cls"], r["defeituosa"]
    boas = cls & (d["tipo"] == "good")
    ruins = cls & (d["tipo"] != "good")
    esp, rec = 1 - det[boas].mean(), det[ruins].mean()
    vp, fp = det[ruins].sum(), det[boas].sum()
    prec = vp / max(vp + fp, 1)
    s = {"especificidade": esp, "recall": rec, "acuracia_balanceada": (esp + rec) / 2,
         "precisao": prec, "f1_defeituosa": 2 * prec * rec / max(prec + rec, 1e-9),
         "roc_auc": auc(r["razao"].max(1)[ruins], r["razao"].max(1)[boas])}
    for t in TIPOS:
        s[f"recall_{t}"] = det[cls & (d["tipo"] == t)].mean()
    return s


def tipo_apontado(r: dict) -> np.ndarray:
    """Tipo apontado = primeira etapa da cascata que acusa; 'good' se nenhuma acusa.

    A ordem da cascata vai do especialista mais específico para o mais geral: amassado e
    impressão quase só acusam o próprio defeito; furinho e arranhão medem contraste local e
    também reagem aos outros defeitos, por isso ficam no fim."""
    ap = np.full(len(r["defeituosa"]), "good", dtype=object)
    for j in reversed(range(len(r["estagios"]))):
        ap[r["disparo"][:, j]] = r["estagios"][j]
    return ap


# =============================================================================
# 3. LOCALIZAÇÃO: máscara da cascata x ground truth
# =============================================================================
def mascara_cascata(arquivo: str, disparo: np.ndarray) -> np.ndarray:
    """União das máscaras das etapas que acusaram."""
    m = None
    for e, d in zip(ESTAGIOS, disparo):
        if d:
            p = cv2.imread(str(DIR_MASCARAS / f"{_chave(arquivo)}__{e}.png"), cv2.IMREAD_GRAYSCALE)
            m = p if m is None else m | p
    return m if m is not None else np.zeros((1000, 1000), np.uint8)


def avaliar_localizacao(d: dict, r: dict) -> str:
    reg = {t: [] for t in ["good"] + TIPOS}
    for i in np.where(r["cls"])[0]:
        t, arq = d["tipo"][i], d["arquivo"][i]
        m = mascara_cascata(arq, r["disparo"][i])
        gt = carregar_mascara_gt(t, int(Path(arq).stem), m.shape)
        reg[t].append(metricas_pixel(m, gt))
    txt = [f"Localização: união das máscaras das etapas que acusaram x ground truth (ALFA = {ALFA})",
           f"hit = fração das defeituosas com >= {MIN_PX_HIT} px marcados dentro do defeito real", "",
           f"{'tipo':<16}{'n':>4}{'hit':>7}{'precisão':>10}{'recall':>8}{'dice':>7}{'iou':>7}"]
    todos = []
    for t in TIPOS:
        ms = reg[t]
        todos += ms
        tp, fp = sum(m["tp"] for m in ms), sum(m["fp"] for m in ms)
        txt.append(f"{t:<16}{len(ms):4d}{np.mean([m['tp'] >= MIN_PX_HIT for m in ms]):7.2f}"
                   f"{tp / (tp + fp + 1e-9):10.2f}{np.mean([m['recall'] for m in ms]):8.2f}"
                   f"{np.mean([m['dice'] for m in ms]):7.3f}{np.mean([m['iou'] for m in ms]):7.3f}")
    tp, fp = sum(m["tp"] for m in todos), sum(m["fp"] for m in todos)
    txt.append(f"{'todas':<16}{len(todos):4d}{np.mean([m['tp'] >= MIN_PX_HIT for m in todos]):7.2f}"
               f"{tp / (tp + fp + 1e-9):10.2f}{np.mean([m['recall'] for m in todos]):8.2f}"
               f"{np.mean([m['dice'] for m in todos]):7.3f}{np.mean([m['iou'] for m in todos]):7.3f}")
    boas = reg["good"]
    txt += ["", f"Boas: {np.mean([m['fp'] for m in boas]):.0f} px falsos por imagem em média; "
                f"{np.mean([m['fp'] > 0 for m in boas]):.0%} das boas com algum pixel marcado"]
    return "\n".join(txt)


# =============================================================================
# 4. TABELAS
# =============================================================================
def tabela_especialistas(d: dict) -> str:
    cls = d["conjunto"] == "classificacao"
    boas = cls & (d["tipo"] == "good")
    txt = ["AUC de cada especialista (linha) para separar cada tipo de defeito (coluna) das boas",
           "(192 boas e 109 defeituosas de 'classificacao'; 0,50 = acaso; * = defeito-alvo)", "",
           f"{'especialista':<18}" + "".join(f"{t:>16}" for t in TIPOS) + f"{'todos':>9}"]
    for e in ESTAGIOS:
        linha = f"{NOMES[e]:<18}"
        for t in TIPOS:
            v = f"{auc(d[e][cls & (d['tipo'] == t)], d[e][boas]):.3f}" + ("*" if t == e else " ")
            linha += f"{v:>16}"
        txt.append(linha + f"{auc(d[e][cls & (d['tipo'] != 'good')], d[e][boas]):9.3f}")
    return "\n".join(txt)


def tabela_cascata(d: dict) -> tuple[str, list[dict]]:
    linhas = []
    for a in ALFAS:
        linhas.append({"alfa": a, **resumo(d, decidir(d, a))})
    txt = ["Cascata dos 5 especialistas, validação cruzada de 5 folds (301 imagens)",
           "ALFA = alarme falso aceito por etapa nas boas de treino", "",
           f"{'ALFA':>5}{'espec.':>8}{'recall':>8}{'ac.bal.':>9}{'precisão':>10}{'F1':>6}{'AUC':>7}  "
           + "".join(f"{t[:7]:>9}" for t in TIPOS)]
    for l in linhas:
        txt.append(f"{l['alfa']:5.2f}{l['especificidade']:8.3f}{l['recall']:8.3f}"
                   f"{l['acuracia_balanceada']:9.3f}{l['precisao']:10.3f}{l['f1_defeituosa']:6.2f}"
                   f"{l['roc_auc']:7.3f}  " + "".join(f"{l['recall_' + t]:9.2f}" for t in TIPOS))
    sem = resumo(d, decidir(d, ALFA, margem=False))
    txt += ["", f"Regra do limiar em ALFA = {ALFA}: só o percentil das boas (sem levar ao meio da "
                f"folga) -> especificidade {sem['especificidade']:.3f}, recall {sem['recall']:.3f}, "
                f"acurácia balanceada {sem['acuracia_balanceada']:.3f}"]
    return "\n".join(txt), linhas


def tabela_disparos(d: dict, r: dict) -> str:
    cls = r["cls"]
    tipos = ["good"] + TIPOS
    n = {t: int((cls & (d["tipo"] == t)).sum()) for t in tipos}
    txt = [f"Quantas cápsulas de cada tipo (coluna) cada etapa (linha) acusa, ALFA = {ALFA}", "",
           f"{'etapa':<18}" + "".join(f"{t + ' (' + str(n[t]) + ')':>20}" for t in tipos)]
    for j, e in enumerate(ESTAGIOS):
        txt.append(f"{NOMES[e]:<18}" + "".join(
            f"{int(r['disparo'][cls & (d['tipo'] == t), j].sum()):>20d}" for t in tipos))
    txt.append(f"{'alguma etapa':<18}" + "".join(
        f"{int(r['defeituosa'][cls & (d['tipo'] == t)].sum()):>20d}" for t in tipos))

    ap = tipo_apontado(r)
    txt += ["", "Tipo apontado (primeira etapa da cascata que acusa) x tipo verdadeiro:", "",
            f"{'verdadeiro':<18}" + "".join(f"{t:>16}" for t in tipos) + "   acerto"]
    acertos = []
    for t in tipos:
        sel = cls & (d["tipo"] == t)
        ac = float(np.mean(ap[sel] == t))
        if t != "good":
            acertos.append(ac)
        txt.append(f"{t:<18}" + "".join(f"{int((ap[sel] == p).sum()):>16d}" for p in tipos)
                   + f"{ac:9.2f}")
    ruins = cls & (d["tipo"] != "good") & r["defeituosa"]
    txt += ["", f"Acurácia balanceada do tipo nas 5 classes de defeito (não achada conta como "
                f"erro): {np.mean(acertos):.2f}",
            f"Entre as defeituosas achadas, tipo certo em {np.mean(ap[ruins] == d['tipo'][ruins]):.0%}"]
    return "\n".join(txt)


def tabela_ablacao(d: dict) -> str:
    txt = [f"Ablação: cascata sem cada etapa (ALFA = {ALFA}) e cada etapa sozinha", "",
           f"{'configuração':<26}{'espec.':>8}{'recall':>8}{'ac.bal.':>9}  "
           + "".join(f"{t[:7]:>9}" for t in TIPOS)]

    def linha(nome, est):
        s = resumo(d, decidir(d, ALFA, est))
        return (f"{nome:<26}{s['especificidade']:8.3f}{s['recall']:8.3f}"
                f"{s['acuracia_balanceada']:9.3f}  " + "".join(f"{s['recall_' + t]:9.2f}" for t in TIPOS))

    txt.append(linha("cascata completa", ESTAGIOS))
    for e in ESTAGIOS:
        txt.append(linha(f"sem {NOMES[e]}", [x for x in ESTAGIOS if x != e]))
    txt.append("")
    for e in ESTAGIOS:
        txt.append(linha(f"só {NOMES[e]}", [e]))
    return "\n".join(txt)


def salvar_erros(d: dict, r: dict) -> tuple[list[int], list[int]]:
    cls, det = r["cls"], r["defeituosa"]
    fp = [i for i in np.where(cls & (d["tipo"] == "good") & det)[0]]
    fn = [i for i in np.where(cls & (d["tipo"] != "good") & ~det)[0]]
    with open(DIR_RESULT / "erros.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["arquivo", "tipo", "erro", "etapas_que_acusaram", "maior_valor_sobre_limiar"])
        for i in fp + fn:
            acusou = "+".join(e for e, x in zip(ESTAGIOS, r["disparo"][i]) if x)
            w.writerow([d["arquivo"][i], d["tipo"][i],
                        "falso positivo" if d["tipo"][i] == "good" else "falso negativo",
                        acusou, round(float(r["razao"][i].max()), 2)])
    return fp, fn


# =============================================================================
# 5. FIGURAS
# =============================================================================
def _eixos(ax):
    ax.set_facecolor(FUNDO)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRADE)
    ax.tick_params(colors=TINTA_2, length=0)
    ax.grid(color=GRADE, linewidth=0.8)
    ax.set_axisbelow(True)


def figura_operacao(linhas: list[dict]) -> None:
    """Especificidade x recall: cascata (vários ALFA) e os dois pontos do método único."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.6, 6), facecolor=FUNDO)
    _eixos(ax)
    x = [100 * l["especificidade"] for l in linhas]
    y = [100 * l["recall"] for l in linhas]
    ax.plot(x, y, color=AZUL, linewidth=2, marker="o", markersize=8, markeredgecolor=FUNDO,
            markeredgewidth=2, label="cascata de especialistas (Etapa 7)")
    for l, xi, yi in zip(linhas, x, y):
        ax.annotate(f"ALFA {l['alfa']:.0%}", (xi, yi), textcoords="offset points", xytext=(-8, -4),
                    ha="right", va="top", fontsize=9, color=TINTA_2)
    unico = [(87, 52, "decisão padrão"), (19, 94, "prioriza defeitos")]
    ax.scatter([u[0] for u in unico], [u[1] for u in unico], s=90, color=LARANJA, marker="s",
               edgecolor=FUNDO, linewidth=2, zorder=3, label="método único (Etapa 5)")
    for xi, yi, nome in unico:
        ax.annotate(nome, (xi, yi), textcoords="offset points", xytext=(10, -4), fontsize=9,
                    color=TINTA_2)
    ax.set_xlim(0, 102)
    ax.set_ylim(40, 102)
    ax.set_xlabel("boas aprovadas - especificidade (%)", color=TINTA_2)
    ax.set_ylabel("defeituosas encontradas - recall (%)", color=TINTA_2)
    ax.set_title("Normal x defeituosa: cascata de especialistas e método único", color=TINTA,
                 loc="left", fontsize=12)
    ax.legend(frameon=False, loc="lower left", labelcolor=TINTA)
    fig.tight_layout()
    fig.savefig(DIR_RESULT / "ponto_de_operacao.png", dpi=110, facecolor=FUNDO)
    plt.close(fig)


def figura_encontradas(d: dict, r: dict) -> None:
    """Barras: defeituosas encontradas por tipo e boas aprovadas."""
    import matplotlib.pyplot as plt

    cls, det = r["cls"], r["defeituosa"]
    nomes, valores, rotulos = [], [], []
    for t in TIPOS:
        sel = cls & (d["tipo"] == t)
        nomes.append(NOMES[t])
        valores.append(100 * det[sel].mean())
        rotulos.append(f"{int(det[sel].sum())} de {int(sel.sum())}")
    boas = cls & (d["tipo"] == "good")
    nomes.append("boas aprovadas")
    valores.append(100 * (1 - det[boas].mean()))
    rotulos.append(f"{int((~det[boas]).sum())} de {int(boas.sum())}")

    fig, ax = plt.subplots(figsize=(9, 4.6), facecolor=FUNDO)
    _eixos(ax)
    ax.grid(axis="x", visible=False)
    barras = ax.bar(nomes, valores, width=0.55, color=AZUL)
    for b, rot in zip(barras, rotulos):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, rot, ha="center",
                fontsize=10, color=TINTA)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% das cápsulas do tipo", color=TINTA_2)
    ax.set_title(f"Cascata (ALFA = {ALFA:.0%}): defeituosas encontradas por tipo e boas aprovadas",
                 color=TINTA, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(DIR_RESULT / "encontradas_por_tipo.png", dpi=110, facecolor=FUNDO)
    plt.close(fig)


def figura_disparos(d: dict, r: dict) -> None:
    """Mapa: fração de cada tipo (coluna) que cada etapa (linha) acusa."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cls = r["cls"]
    tipos = ["good"] + TIPOS
    m = np.array([[r["disparo"][cls & (d["tipo"] == t), j].mean() for t in tipos]
                  for j in range(len(ESTAGIOS))])
    mapa = LinearSegmentedColormap.from_list("azul", [FUNDO, AZUL])
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=FUNDO)
    ax.imshow(m, cmap=mapa, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{100 * m[i, j]:.0f}%", ha="center", va="center", fontsize=11,
                    color="white" if m[i, j] > 0.55 else TINTA)
    ax.set_xticks(range(len(tipos)))
    ax.set_xticklabels(["boa"] + [NOMES[t] for t in TIPOS], color=TINTA)
    ax.set_yticks(range(len(ESTAGIOS)))
    ax.set_yticklabels([f"{k}. {NOMES[e]}" for k, e in enumerate(ESTAGIOS, 1)], color=TINTA)
    ax.set_xlabel("tipo verdadeiro da cápsula", color=TINTA_2)
    ax.set_ylabel("etapa da cascata", color=TINTA_2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"Que etapa acusa cada tipo de cápsula (ALFA = {ALFA:.0%})", color=TINTA,
                 loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(DIR_RESULT / "disparos_por_tipo.png", dpi=110, facecolor=FUNDO)
    plt.close(fig)


def figuras_exemplos(d: dict, r: dict, lim: dict[str, float], fp: list[int], fn: list[int]) -> None:
    """Passo a passo de um exemplo por tipo e dos erros mais claros."""
    refs = referencias_do_banco()
    exemplos = ["test/squeeze/002.png", "test/faulty_imprint/000.png", "test/crack/010.png",
                "test/poke/001.png", "test/scratch/009.png", "test/good/005.png"]
    for nome, lista in [("exemplo", exemplos),
                        ("erro_falso_positivo", [d["arquivo"][i] for i in fp[:2]]),
                        ("erro_falso_negativo", [d["arquivo"][i] for i in fn[:2]])]:
        for arq in lista:
            tipo, indice = arq.split("/")[1], int(Path(arq).stem)
            bgr = carregar_imagem(DATASET_DIR / arq)
            gt = carregar_mascara_gt(tipo, indice, bgr.shape[:2])
            figura_passo_a_passo(bgr, gt, refs, arq[:-4], limiares=lim, mostrar=False,
                                 salvar=DIR_RESULT / f"{nome}_{tipo}_{Path(arq).parts[0]}_{indice:03d}.png")


# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 7: cascata de especialistas")
    parser.add_argument("--recalcular", action="store_true",
                        help="refaz os valores dos especialistas (dados/especialistas.csv)")
    parser.add_argument("--sem-figuras", action="store_true")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    DIR_RESULT.mkdir(parents=True, exist_ok=True)
    if args.recalcular or not CSV_VALORES.exists() or not DIR_MASCARAS.exists():
        calcular_valores()
    d = carregar_valores()

    def salvar(nome: str, texto: str) -> None:
        (DIR_RESULT / nome).write_text(texto + "\n", encoding="utf-8")
        print("\n" + texto)

    salvar("tabela_especialistas.txt", tabela_especialistas(d))
    texto, linhas = tabela_cascata(d)
    salvar("tabela_cascata.txt", texto)
    with open(DIR_RESULT / "tabela_cascata.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()), lineterminator="\n")
        w.writeheader()
        for l in linhas:
            w.writerow({k: round(float(v), 4) for k, v in l.items()})

    r = decidir(d, ALFA)
    salvar("disparos_por_tipo.txt", tabela_disparos(d, r))
    salvar("ablacao.txt", tabela_ablacao(d))
    salvar("localizacao.txt", avaliar_localizacao(d, r))
    fp, fn = salvar_erros(d, r)
    print(f"\nErros (ALFA = {ALFA}): {len(fp)} boas rejeitadas e {len(fn)} defeituosas aprovadas "
          f"-> {DIR_RESULT / 'erros.csv'}")

    # Limiares finais, com todas as imagens rotuladas (para usar em imagens novas)
    todas = np.ones(len(d["tipo"]), bool)
    finais = {f"alfa_{a}": limiares(d, todas, a) for a in (ALFA, ALFA_PRIORIZA)}
    JSON_LIMIARES.write_text(json.dumps(finais, indent=2) + "\n", encoding="utf-8")
    print("Limiares finais (todas as imagens), ALFA =", ALFA, ":",
          {NOMES[e]: round(v, 3) for e, v in finais[f"alfa_{ALFA}"].items()})

    if not args.sem_figuras:
        figura_operacao(linhas)
        figura_encontradas(d, r)
        figura_disparos(d, r)
        figuras_exemplos(d, r, finais[f"alfa_{ALFA}"], fp, fn)
    print(f"\nResultados em {DIR_RESULT}")


if __name__ == "__main__":
    main()
