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
    dados/especialistas.csv                     valor de cada especialista por imagem (e, na
                                                coluna poke_anterior, o do método anterior do
                                                furinho, que não entra na cascata)
    dados/limiares_cascata.json                 limiares finais (todas as imagens), para uso
    dados/gerado/especialistas/*.png            máscaras de cada especialista (não versionado)
    resultados/etapa7/tabela_especialistas.txt  AUC de cada especialista x tipo de defeito
    resultados/etapa7/tabela_cascata.txt|csv    cascata em vários pontos de operação (ALFA)
    resultados/etapa7/disparos_por_tipo.txt     que etapa acusa cada tipo; tipo apontado
    resultados/etapa7/ablacao.txt               cascata sem cada etapa
    resultados/etapa7/furinho_antes_depois.txt  furinho: top-hat (anterior) x LoG + circularidade
    resultados/etapa7/furinho_log_otsu.png      passo a passo do especialista do furinho
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
    CHAVE_FRACAO_CLARA, ESTAGIOS, FURINHO_AREA_MIN, NOMES, blobs_furinho, candidato_furinho_tophat,
    diferenca_do_500, examinar, figura_passo_a_passo, preparar, referencias_do_banco,
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


COLUNA_ANTERIOR = "poke_anterior"   # valor do método anterior do furinho (top-hat 15 px)
COLUNA_CLARA = CHAVE_FRACAO_CLARA   # fração clara da diferença do "500" (refina o tipo)
EXTRAS = [COLUNA_ANTERIOR, COLUNA_CLARA]


def _examinar_linha(l: dict) -> dict:
    bgr = carregar_imagem(l["caminho"])
    ctx = preparar(bgr)
    valores, mascaras = examinar(bgr, _REFS, ctx)
    for nome, m in mascaras.items():
        cv2.imwrite(str(DIR_MASCARAS / f"{_chave(l['arquivo'])}__{nome}.png"), m)
    # Só para a tabela "antes x depois": o método anterior do furinho não entra na cascata
    valores[COLUNA_ANTERIOR] = candidato_furinho_tophat(ctx, _REFS)[0]
    # Não entra na decisão normal x defeituosa: só ajuda a dizer o tipo quando a etapa 2 acusa
    valores[COLUNA_CLARA] = diferenca_do_500(ctx, _REFS)[COLUNA_CLARA]
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
              **{e: round(v[e], 5) for e in ESTAGIOS + EXTRAS}}
             for l, v in zip(linhas, valores)]
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
    for e in ESTAGIOS + EXTRAS:
        if e in linhas[0]:
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


def limiar_fracao_clara(d: dict, treino: np.ndarray, limiar_impressao: float) -> float:
    """Quando a etapa 2 acusa: é impressão falha ou um furinho em cima do "500"?

    Na impressão falha FALTA tinta: a cápsula fica mais escura que o modelo e a fração clara da
    diferença é ~0. Um furo sobre a impressão expõe material mais claro. O corte é o que melhor
    separa, no TREINO, as impressões falhas dos furinhos que a etapa 2 acusa (ponto médio entre
    dois valores vizinhos; soma do acerto das duas classes). Sem exemplos das duas classes, o
    corte é 1 (a fração nunca passa de 1, então nada é renomeado)."""
    if COLUNA_CLARA not in d:
        return 1.0
    acusou = treino & (d["faulty_imprint"] > limiar_impressao)
    falha = d[COLUNA_CLARA][acusou & (d["tipo"] == "faulty_imprint")]
    furo = d[COLUNA_CLARA][acusou & (d["tipo"] == "poke")]
    valores = np.unique(np.concatenate([falha, furo]))
    if len(falha) == 0 or len(furo) == 0 or len(valores) < 2:
        return 1.0
    cortes = (valores[:-1] + valores[1:]) / 2
    acerto = [np.mean(falha <= c) + np.mean(furo > c) for c in cortes]
    return float(cortes[int(np.argmax(acerto))])


def limiares(d: dict, treino: np.ndarray, alfa: float, margem: bool = True) -> dict[str, float]:
    """Limiar de cada etapa e, na chave COLUNA_CLARA, o corte que refina o tipo da etapa 2."""
    boas = treino & (d["tipo"] == "good")
    saida = {}
    for e in ESTAGIOS:
        alvo = d[e][treino & (d["tipo"] == e)] if margem else np.array([])
        saida[e] = limiar_etapa(d[e][boas], alvo, alfa)
    saida[COLUNA_CLARA] = limiar_fracao_clara(d, treino, saida["faulty_imprint"])
    return saida


def decidir(d: dict, alfa: float, estagios: list[str] = ESTAGIOS, margem: bool = True) -> dict:
    """Validação cruzada: limiares no treino de cada rodada, decisão no fold de teste.

    Devolve, para as imagens de 'classificacao': disparo[n, etapa] (bool), razao[n, etapa]
    (valor / limiar), defeituosa[n] e furo_no_500[n] (a diferença do "500" é clara: se a etapa
    2 for a primeira a acusar, o tipo apontado é furinho, e não impressão falha)."""
    n = len(d["tipo"])
    cls = d["conjunto"] == "classificacao"
    razao = np.zeros((n, len(estagios)))
    furo_no_500 = np.zeros(n, bool)
    for k in range(5):
        treino = (d["conjunto"] == "calibracao") | (cls & (d["fold"] != k))
        teste = cls & (d["fold"] == k)
        lim = limiares(d, treino, alfa, margem)
        for j, e in enumerate(estagios):
            razao[teste, j] = d[e][teste] / max(lim[e], 1e-9)
        if COLUNA_CLARA in d:
            furo_no_500[teste] = d[COLUNA_CLARA][teste] > lim[COLUNA_CLARA]
    disparo = razao > 1
    return {"disparo": disparo, "razao": razao, "defeituosa": disparo.any(1), "cls": cls,
            "estagios": estagios, "furo_no_500": furo_no_500}


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


def tipo_apontado(r: dict, refinar: bool = True) -> np.ndarray:
    """Tipo apontado = primeira etapa da cascata que acusa; 'good' se nenhuma acusa.

    refinar: quando a primeira etapa a acusar é a da impressão e a diferença do "500" para o
    modelo é CLARA (r["furo_no_500"]), o tipo é furinho: a impressão falha perde tinta e fica
    mais escura; quem clareia a impressão é um furo em cima dela. refinar=False é a regra
    simples (só a ordem das etapas).

    A ordem da cascata vai do especialista mais específico para o mais geral: amassado e
    impressão quase só acusam o próprio defeito; o furinho acusa qualquer mancha compacta
    (também as lascas da rachadura) e o arranhão mede contraste local fino e reage a quase
    todos os defeitos, por isso ficam no fim."""
    ap = np.full(len(r["defeituosa"]), "good", dtype=object)
    for j in reversed(range(len(r["estagios"]))):
        ap[r["disparo"][:, j]] = r["estagios"][j]
    if refinar and "poke" in r["estagios"]:
        ap[(ap == "faulty_imprint") & r["furo_no_500"]] = "poke"
    return ap


def f1_macro_tipos(verdadeiro: np.ndarray, apontado: np.ndarray) -> float:
    """Média do F1 dos 5 tipos de defeito. Boa acusada conta como falso positivo do tipo
    apontado; defeituosa não achada, como falso negativo do tipo verdadeiro."""
    f1 = []
    for t in TIPOS:
        vp = np.sum((verdadeiro == t) & (apontado == t))
        f1.append(2 * vp / max(np.sum(verdadeiro == t) + np.sum(apontado == t), 1))
    return float(np.mean(f1))


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

    ruins = cls & (d["tipo"] != "good") & r["defeituosa"]
    for titulo, refinar in (
            ("Tipo apontado x tipo verdadeiro. Regra: primeira etapa que acusa; se for a da "
             "impressão e a\ndiferença do \"500\" para o modelo for clara, o tipo é furinho "
             "(corte aprendido no treino):", True),
            ("Regra simples, sem olhar a diferença do \"500\" (só a primeira etapa que acusa):",
             False)):
        ap = tipo_apontado(r, refinar)
        txt += ["", titulo, "",
                f"{'verdadeiro':<18}" + "".join(f"{t:>16}" for t in tipos) + "   acerto"]
        acertos = []
        for t in tipos:
            sel = cls & (d["tipo"] == t)
            ac = float(np.mean(ap[sel] == t))
            if t != "good":
                acertos.append(ac)
            txt.append(f"{t:<18}" + "".join(f"{int((ap[sel] == p).sum()):>16d}" for p in tipos)
                       + f"{ac:9.2f}")
        txt += ["", f"Acurácia balanceada do tipo nas 5 classes de defeito (não achada conta "
                    f"como erro): {np.mean(acertos):.2f}",
                f"F1 macro dos 5 tipos: {f1_macro_tipos(d['tipo'][cls], ap[cls]):.2f}",
                "Entre as defeituosas achadas, tipo certo em "
                f"{np.mean(ap[ruins] == d['tipo'][ruins]):.0%}"]
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


def tabela_furinho(d: dict) -> str:
    """Etapa do furinho antes (top-hat 15 px, área da maior mancha) x depois (LoG + Otsu no
    círculo + circularidade): a mesma cascata, trocando só o valor da etapa 4."""
    antes = dict(d)
    antes["poke"] = d[COLUNA_ANTERIOR]
    cls = d["conjunto"] == "classificacao"
    boas = cls & (d["tipo"] == "good")
    tipos = ["good"] + TIPOS
    j = ESTAGIOS.index("poke")
    txt = ["Etapa do furinho: método anterior x LoG + Otsu no círculo + circularidade",
           "anterior = top-hat/black-hat 15 px acima do envelope, área da maior mancha",
           "depois   = resposta do LoG (3 escalas, acima do envelope) x circularidade da região do "
           f"Otsu no círculo, área >= {FURINHO_AREA_MIN} px",
           "tipo = primeira etapa que acusa (regra simples, sem o refinamento pela diferença "
           "do \"500\")", "",
           "AUC do valor da etapa para separar cada tipo das boas (0,50 = não reage ao tipo):",
           f"{'':<10}" + "".join(f"{t:>16}" for t in TIPOS)]
    for nome, dd in (("anterior", antes), ("depois", d)):
        txt.append(f"{nome:<10}" + "".join(
            f"{auc(dd['poke'][cls & (d['tipo'] == t)], dd['poke'][boas]):16.3f}" for t in TIPOS))
    for alfa in (ALFA, ALFA_PRIORIZA):
        txt += ["", f"ALFA = {alfa}",
                f"{'':<10}{'espec.':>8}{'recall':>8}{'ac.bal.':>9}{'AUC':>7}{'tipo':>7}   "
                "acerto do tipo: " + " / ".join(t[:7] for t in TIPOS)]
        linhas_etapa, linhas_sem = [], []
        for nome, dd in (("anterior", antes), ("depois", d)):
            r = decidir(dd, alfa)
            s = resumo(dd, r)
            ap = tipo_apontado(r, refinar=False)
            ac = [float(np.mean(ap[cls & (d["tipo"] == t)] == t)) for t in TIPOS]
            txt.append(f"{nome:<10}{s['especificidade']:8.3f}{s['recall']:8.3f}"
                       f"{s['acuracia_balanceada']:9.3f}{s['roc_auc']:7.3f}{np.mean(ac):7.2f}   "
                       + " / ".join(f"{a:.2f}" for a in ac))
            linhas_etapa.append(f"{nome:<10}" + "".join(
                f"{int(r['disparo'][cls & (d['tipo'] == t), j].sum()):>16d}" for t in tipos))
            sem = decidir(dd, alfa, [e for e in ESTAGIOS if e != "poke"])
            so_ela = r["defeituosa"] & ~sem["defeituosa"] & cls
            linhas_sem.append(f"{nome:<10}" + "".join(
                f"{int((so_ela & (d['tipo'] == t)).sum()):>16d}" for t in tipos))
        txt += ["", "  quantas cápsulas de cada tipo a etapa do furinho acusa:",
                f"{'':<10}" + "".join(f"{t:>16}" for t in tipos)] + linhas_etapa
        txt += ["", "  quantas só a etapa do furinho acusa (perdidas se ela for retirada):",
                f"{'':<10}" + "".join(f"{t:>16}" for t in tipos)] + linhas_sem
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


def figura_furinho(refs) -> None:
    """Passo a passo do especialista do furinho em 5 cápsulas: o que o método anterior marcava,
    o LoG acima do envelope, a região do Otsu no círculo e a circularidade."""
    import matplotlib.pyplot as plt

    # (arquivo, comentário, onde centrar o recorte): "log" = blob que dá o valor; "pico" = maior
    # pico do LoG, mesmo sem área mínima; "anterior" = maior mancha do método anterior
    exemplos = [("test/poke/016.png", "furinho na tampa preta", "log"),
                ("test/poke/014.png", "furinho na metade laranja", "log"),
                ("test/scratch/008.png", "arranhão", "anterior"),
                ("test/faulty_imprint/010.png", "impressão falha", "anterior"),
                ("train/good/130.png", "boa com ponto de poeira", "pico")]
    meio = 90
    fig, eixos = plt.subplots(len(exemplos), 4, figsize=(13.5, 3.3 * len(exemplos)), facecolor=FUNDO)
    for linha, (arq, nome, centro) in zip(eixos, exemplos):
        bgr = carregar_imagem(DATASET_DIR / arq)
        ctx = preparar(bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        area_ant, masc_ant = candidato_furinho_tophat(ctx, refs)
        mapa, blobs = blobs_furinho(ctx, refs)
        b = max(blobs, key=lambda b: (b["valor"], b["resposta"]))
        if centro == "pico":
            b = max(blobs, key=lambda b: b["resposta"])
        if centro == "anterior":
            n, _, stats, cent = cv2.connectedComponentsWithStats(masc_ant, connectivity=8)
            cx, cy = cent[1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))]
            perto = [k for k in blobs if abs(k["x"] - cx) < meio and abs(k["y"] - cy) < meio]
            if perto:
                b = max(perto, key=lambda k: (k["valor"], k["resposta"]))
        if centro != "anterior":
            cx, cy = b["x"], b["y"]
        x0 = int(np.clip(cx - meio, 0, bgr.shape[1] - 2 * meio))
        y0 = int(np.clip(cy - meio, 0, bgr.shape[0] - 2 * meio))
        rec = lambda im: im[y0:y0 + 2 * meio, x0:x0 + 2 * meio]

        linha[0].imshow(rec(rgb))
        linha[0].set_title(f"{arq[:-4]}\n{nome}", fontsize=10, color=TINTA)
        ant = (rgb * 0.6).astype(np.uint8)
        ant[masc_ant > 0] = (255, 60, 60)
        linha[1].imshow(rec(ant))
        linha[1].set_title(f"anterior: top-hat/black-hat 15 px\nmaior mancha = {area_ant:.0f} px",
                           fontsize=10, color=TINTA)
        linha[2].imshow(rec(mapa), cmap="magma", vmin=0, vmax=max(20.0, float(rec(mapa).max())))
        linha[2].set_title(f"LoG acima do envelope\npico = {b['resposta']:.1f}  "
                           f"(sigma = {b['sigma']} px)", fontsize=10, color=TINTA)
        seg = rgb.copy()
        seg[b["regiao"] > 0] = (0.45 * seg[b["regiao"] > 0] + 0.55 * np.array([0, 229, 255])).astype(np.uint8)
        linha[3].imshow(rec(seg))
        linha[3].add_patch(plt.Circle((b["x"] - x0, b["y"] - y0), b["raio"], fill=False,
                                      color="white", linewidth=1.2))
        pequeno = b["area"] < FURINHO_AREA_MIN
        linha[3].set_title(f"Otsu no círculo: área {b['area']:.0f} px"
                           + (f" (< {FURINHO_AREA_MIN})" if pequeno else "")
                           + f"\ncircularidade {b['circularidade']:.2f}  ->  valor = {b['valor']:.1f}",
                           fontsize=10, color=TINTA)
        for ax in linha:
            ax.axis("off")
    fig.suptitle("Especialista do furinho: Laplaciano da Gaussiana, Otsu no círculo e circularidade",
                 fontsize=13, color=TINTA, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(DIR_RESULT / "furinho_log_otsu.png", dpi=90, facecolor=FUNDO)
    plt.close(fig)


def figuras_exemplos(d: dict, r: dict, lim: dict[str, float], fp: list[int], fn: list[int]) -> None:
    """Passo a passo de um exemplo por tipo e dos erros mais claros."""
    refs = referencias_do_banco()
    figura_furinho(refs)
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
    if any(c not in d for c in EXTRAS):
        print(f"Aviso: {CSV_VALORES.name} não tem as colunas {EXTRAS}; rode com --recalcular "
              "para gerar furinho_antes_depois.txt e refinar o tipo pela diferença do \"500\"")

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
    if COLUNA_ANTERIOR in d:
        salvar("furinho_antes_depois.txt", tabela_furinho(d))
    salvar("localizacao.txt", avaliar_localizacao(d, r))
    fp, fn = salvar_erros(d, r)
    print(f"\nErros (ALFA = {ALFA}): {len(fp)} boas rejeitadas e {len(fn)} defeituosas aprovadas "
          f"-> {DIR_RESULT / 'erros.csv'}")

    # Limiares finais, com todas as imagens rotuladas (para usar em imagens novas)
    todas = np.ones(len(d["tipo"]), bool)
    finais = {f"alfa_{a}": limiares(d, todas, a) for a in (ALFA, ALFA_PRIORIZA)}
    JSON_LIMIARES.write_text(json.dumps(finais, indent=2) + "\n", encoding="utf-8")
    print("Limiares finais (todas as imagens), ALFA =", ALFA, ":",
          {NOMES[e]: round(finais[f"alfa_{ALFA}"][e], 3) for e in ESTAGIOS},
          "| fração clara do \"500\":", round(finais[f"alfa_{ALFA}"][COLUNA_CLARA], 3))

    if not args.sem_figuras:
        figura_operacao(linhas)
        figura_encontradas(d, r)
        figura_disparos(d, r)
        figuras_exemplos(d, r, finais[f"alfa_{ALFA}"], fp, fn)
    print(f"\nResultados em {DIR_RESULT}")


if __name__ == "__main__":
    main()
