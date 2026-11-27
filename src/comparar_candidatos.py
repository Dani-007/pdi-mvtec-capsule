"""Etapa 7: compara os métodos candidatos de cada defeito (AUC e achadas) e o efeito da
morfologia em cada especialista.

Uso:  python src/comparar_candidatos.py   (~4 min) -> tabela_candidatos.txt, tabela_morfologia.txt
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

import cascata
from cascata import DIR_RESULT, TIPOS, auc, mapear
from especialistas import (
    NOMES, TOPHAT_FURINHO_K, _centroide, candidato_arranhao_canny, candidato_arranhao_maior_risco,
    candidato_furinho_tophat, candidato_impressao_area, candidato_squeeze_descritores,
    construir_referencias, especialista_arranhao, especialista_furinho, especialista_impressao,
    especialista_rachadura, especialista_squeeze, preparar, segmentar_laranja,
)
from particao import carregar_particao
from pipeline import RAIZ, carregar_imagem, carregar_mascara_gt, metricas_pixel

K_GRANDE = 25
MIN_PX_HIT = cascata.MIN_PX_HIT
CIRC_MIN_TOPHAT = 0.4   # candidato "circularidade sem o LoG" (4 pi A / P² de cada mancha)
ALFA_COMPARACAO = 0.02

# (defeito-alvo, chave, descrição, escolhido?)
CANDIDATOS = [
    ("squeeze", "sq_descritores", "descritores globais da silhueta (maior z-score)", False),
    ("squeeze", "sq_xor", "XOR com a silhueta boa mais parecida, sem abertura", False),
    ("squeeze", "sq_xor_abertura", "XOR com a silhueta boa mais parecida + abertura 7", True),
    ("faulty_imprint", "imp_area", 'desvio da área do "500" (atributo da Etapa 4)', False),
    ("faulty_imprint", "imp_inteiro", 'correlação do "500" inteiro com o melhor modelo', False),
    ("faulty_imprint", "imp_algarismo", "correlação do pior algarismo com o melhor modelo", True),
    ("crack", "cr_cinza", "forma da tampa preta, Otsu no cinza", False),
    ("crack", "cr_kmeans", "forma da tampa preta, k-means (2 grupos) em a*, b*", False),
    ("crack", "cr_a", "forma da tampa preta, Otsu no a* (Lab)", True),
    ("crack", "fu_env15", "contraste local 15 px (método anterior do furinho)", False),
    ("crack", "fu_log_forma", "blobs do LoG + circularidade (o especialista do furinho)", False),
    ("poke", "fu_sem_env", "top-hat/black-hat 15 px, sem envelope", False),
    ("poke", "fu_env25", "top-hat/black-hat 25 px, acima do envelope", False),
    ("poke", "fu_env15", "top-hat/black-hat 15 px, acima do envelope (anterior)", False),
    ("poke", "fu_env15_circ", "anterior, só manchas com circularidade >= 0,4", False),
    ("poke", "fu_log", "LoG em 3 escalas, acima do envelope (maior pico)", False),
    ("poke", "fu_log_regiao", "LoG + crescimento de região + circularidade", False),
    ("poke", "fu_log_forma", "LoG + Otsu no círculo + circularidade", True),
    ("scratch", "ar_canny", "pixels de borda do Canny (Etapa 1)", False),
    ("scratch", "ar_maior_sem_env", "maior risco (top-hat 9 px), sem envelope", False),
    ("scratch", "ar_maior", "maior risco (top-hat 9 px), acima do envelope", False),
    ("scratch", "ar_dens_sem_env", "concentração de fragmentos, sem envelope", False),
    ("scratch", "ar_dens", "concentração de fragmentos, acima do envelope", True),
]

# Morfologia de cada especialista: (defeito-alvo, descrição sem, descrição com, sem, com)
MORFOLOGIA = [
    ("squeeze", "XOR das silhuetas", "+ abertura 7x7",
     lambda c, r: especialista_squeeze(c, r, abertura=1), especialista_squeeze),
    ("faulty_imprint", "|teste - modelo| > 30", "+ abertura 3x3",
     lambda c, r: especialista_impressao(c, r, abrir=False), especialista_impressao),
    ("crack", "XOR das tampas", "+ abertura 9x9",
     lambda c, r: especialista_rachadura(c, r, abertura=1), especialista_rachadura),
    ("poke", "Otsu no círculo", "+ fechamento 5 e abertura 3",
     lambda c, r: especialista_furinho(c, r, morfologia=False), especialista_furinho),
    ("scratch", "fragmentos do top-hat", "+ remoção < 4 px e fechamento 15",
     lambda c, r: especialista_arranhao(c, r, morfologia=False), especialista_arranhao),
]

_TAMPAS = {}   # tampas pretas do banco por segmentação alternativa (candidatos cr_cinza/kmeans)


def _tampa(ctx, canal: str) -> np.ndarray:
    laranja = segmentar_laranja(ctx.bgr, ctx.capsula, canal=canal)
    return np.where((ctx.capsula > 0) & (laranja == 0), 255, 0).astype(np.uint8)


def _morfologia(ctx, refs, l: dict) -> dict:
    """Para cada especialista, sem e com morfologia: (valor, pixels marcados, métricas pixel a
    pixel contra o ground truth, só quando a imagem é do defeito-alvo)."""
    saida = {}
    for alvo, _, _, sem, com in MORFOLOGIA:
        gt = (carregar_mascara_gt(alvo, int(Path(l["arquivo"]).stem), ctx.capsula.shape)
              if l["tipo"] == alvo else None)
        for chave, funcao in (("sem", sem), ("com", com)):
            valor, mascara = funcao(ctx, refs)
            saida[(alvo, chave)] = (valor, cv2.countNonZero(mascara),
                                    metricas_pixel(mascara, gt) if gt is not None else None)
    return saida


def _valores(l: dict) -> dict:
    refs = cascata.referencias()
    ctx = preparar(carregar_imagem(l["caminho"]))
    v = {}
    v["sq_descritores"] = candidato_squeeze_descritores(ctx, refs)
    v["sq_xor"] = especialista_squeeze(ctx, refs, abertura=1)[0]
    v["sq_xor_abertura"] = especialista_squeeze(ctx, refs)[0]
    v["imp_area"] = candidato_impressao_area(ctx, refs)
    v["imp_inteiro"] = especialista_impressao(ctx, refs, por_algarismo=False)[0]
    v["imp_algarismo"] = especialista_impressao(ctx, refs)[0]
    v["cr_cinza"] = especialista_rachadura(ctx, refs, tampa=_tampa(ctx, "cinza"),
                                           tampas_ref=_TAMPAS["cinza"])[0]
    v["cr_kmeans"] = especialista_rachadura(ctx, refs, tampa=_tampa(ctx, "kmeans"),
                                            tampas_ref=_TAMPAS["kmeans"])[0]
    v["cr_a"] = especialista_rachadura(ctx, refs)[0]
    v["fu_sem_env"] = candidato_furinho_tophat(ctx, refs, usar_envelope=False)[0]
    v["fu_env25"] = candidato_furinho_tophat(ctx, refs, k=K_GRANDE)[0]
    v["fu_env15"] = candidato_furinho_tophat(ctx, refs)[0]
    v["fu_env15_circ"] = candidato_furinho_tophat(ctx, refs, circ_min=CIRC_MIN_TOPHAT)[0]
    v["fu_log"] = especialista_furinho(ctx, refs, usar_forma=False)[0]
    v["fu_log_regiao"] = especialista_furinho(ctx, refs, segmentacao="regiao")[0]
    v["fu_log_forma"] = especialista_furinho(ctx, refs)[0]
    v["ar_canny"] = candidato_arranhao_canny(ctx)
    v["ar_maior_sem_env"] = candidato_arranhao_maior_risco(ctx, refs, usar_envelope=False)
    v["ar_maior"] = candidato_arranhao_maior_risco(ctx, refs)
    v["ar_dens_sem_env"] = especialista_arranhao(ctx, refs, usar_envelope=False)[0]
    v["ar_dens"] = especialista_arranhao(ctx, refs)[0]
    v["morfologia"] = _morfologia(ctx, refs, l)
    return v


def tabela_morfologia(valores: list[dict], tipo: np.ndarray, boas: np.ndarray, cal: np.ndarray,
                      fold: np.ndarray) -> tuple[str, list[dict]]:
    txt = ["Efeito das operações morfológicas de cada especialista (máscara sem limiar de decisão)",
           "nas boas (192): FP px = pixels marcados por cápsula (todos falsos); marcadas = fração "
           "das boas com algum pixel",
           f"no defeito-alvo: hit = >= {MIN_PX_HIT} px dentro do ground truth; precisão, recall, "
           "Dice e IoU pixel a pixel;",
           "AUC e achadas = valor do especialista, alvo x boas (como na tabela de candidatos)", "",
           f"{'especialista':<17}{'operação':<34}{'FP px':>8}{'marcadas':>10}{'hit':>6}"
           f"{'precisão':>10}{'recall':>8}{'Dice':>7}{'IoU':>7}{'AUC':>7}{'achadas':>9}"]
    linhas = []
    for alvo, desc_sem, desc_com, _, _ in MORFOLOGIA:
        for chave, desc in (("sem", desc_sem), ("com", desc_com)):
            dados = [v["morfologia"][(alvo, chave)] for v in valores]
            valor = np.array([x[0] for x in dados])
            px = np.array([x[1] for x in dados])
            ms = [x[2] for x in dados if x[2] is not None]
            tp, fp = sum(m["tp"] for m in ms), sum(m["fp"] for m in ms)
            l = {"especialista": NOMES[alvo], "operacao": desc,
                 "fp_px_boa": float(px[boas].mean()), "boas_marcadas": float((px[boas] > 0).mean()),
                 "hit": float(np.mean([m["tp"] >= MIN_PX_HIT for m in ms])),
                 "precisao": tp / (tp + fp + 1e-9),
                 "recall": float(np.mean([m["recall"] for m in ms])),
                 "dice": float(np.mean([m["dice"] for m in ms])),
                 "iou": float(np.mean([m["iou"] for m in ms])),
                 "auc": auc(valor[tipo == alvo], valor[boas]),
                 "achadas": achadas_cv(valor, tipo == alvo, boas, cal, fold)}
            linhas.append(l)
            txt.append(f"{l['especialista']:<17}{desc:<34}{l['fp_px_boa']:8.0f}"
                       f"{l['boas_marcadas']:10.0%}{l['hit']:6.2f}{l['precisao']:10.2f}"
                       f"{l['recall']:8.2f}{l['dice']:7.3f}{l['iou']:7.3f}{l['auc']:7.3f}"
                       f"{l['achadas']:9.2f}")
        txt.append("")
    return "\n".join(txt), linhas


def acima_cv(valor: np.ndarray, boas: np.ndarray, treino_extra: np.ndarray,
             fold: np.ndarray) -> np.ndarray:
    """Quais imagens ficam acima do percentil 98 das boas de treino, decidido fold a fold."""
    achou = np.zeros(len(valor), bool)
    for k in range(5):
        treino = treino_extra | (boas & (fold != k))
        limiar = np.percentile(valor[treino], 100 * (1 - ALFA_COMPARACAO))
        achou[fold == k] = valor[fold == k] > limiar
    return achou


def achadas_cv(valor: np.ndarray, alvo: np.ndarray, boas: np.ndarray, treino_extra: np.ndarray,
               fold: np.ndarray) -> float:
    """Fração do alvo acima do percentil 98 das boas de treino."""
    return float(acima_cv(valor, boas, treino_extra, fold)[alvo].mean())


def metodo_unico() -> dict[str, tuple[float, float]] | None:
    """Método único (Etapas 3-4): área da maior mancha suspeita, de dados/atributos.csv."""
    caminho = RAIZ / "dados" / "atributos.csv"
    if not caminho.exists():
        return None
    with open(caminho, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    tipo = np.array([l["tipo"] for l in linhas])
    fold = np.array([int(l["fold"]) for l in linhas])
    valor = np.array([float(l["area_maior"]) for l in linhas])
    boas = tipo == "good"
    nada = np.zeros(len(tipo), bool)
    return {t: (auc(valor[tipo == t], valor[boas]), achadas_cv(valor, tipo == t, boas, nada, fold))
            for t in TIPOS}


def main() -> None:
    DIR_RESULT.mkdir(parents=True, exist_ok=True)
    banco = [l["caminho"] for l in carregar_particao("banco")]
    print("Montando as referências com as 30 boas do banco...")
    refs = construir_referencias(banco, ks_claro=(9, TOPHAT_FURINHO_K, K_GRANDE),
                                 ks_escuro=(TOPHAT_FURINHO_K, K_GRANDE))
    _TAMPAS.update({"cinza": [], "kmeans": []})
    for c in banco:
        ctx = preparar(carregar_imagem(c))
        for canal in _TAMPAS:
            t = _tampa(ctx, canal)
            _TAMPAS[canal].append((t, _centroide(t)))

    linhas = carregar_particao("calibracao") + carregar_particao("classificacao")
    print(f"Calculando {len(CANDIDATOS)} candidatos em {len(linhas)} imagens...")
    valores = mapear(_valores, linhas, refs)

    tipo = np.array([l["tipo"] for l in linhas])
    cal = np.array([l["conjunto"] == "calibracao" for l in linhas])
    fold = np.array([-1 if l["fold"] is None else l["fold"] for l in linhas])
    boas = (tipo == "good") & ~cal
    unico = metodo_unico()

    tabela = []
    txt = ["Candidatos por defeito: defeito-alvo x cápsulas boas de 'classificacao'",
           "achadas = fração do defeito-alvo acima do percentil 98 das boas de treino (5 folds)",
           "* = método escolhido (especialistas.py)", ""]
    for alvo in ["squeeze", "faulty_imprint", "crack", "poke", "scratch"]:
        titulo = f"{NOMES[alvo].upper()} ({alvo}, {int((tipo == alvo).sum())} cápsulas)"
        txt.append(f"{titulo:<64}{'AUC':>6}{'achadas':>10}")
        if unico:
            a, r = unico[alvo]
            txt.append(f"    {'método único: maior mancha do golden template (Etapas 3-4)':<60}"
                       f"{a:6.3f}{r:10.2f}")
            tabela.append({"defeito": alvo, "metodo": "metodo unico (Etapas 3-4)", "escolhido": 0,
                           "auc": round(a, 4), "achadas": round(r, 4)})
        for d, chave, descricao, escolhido in CANDIDATOS:
            if d != alvo:
                continue
            v = np.array([x[chave] for x in valores])
            a = auc(v[tipo == alvo], v[boas])
            r = achadas_cv(v, tipo == alvo, boas, cal, fold)
            txt.append(f"  {'*' if escolhido else ' '} {descricao:<60}{a:6.3f}{r:10.2f}")
            tabela.append({"defeito": alvo, "metodo": descricao, "escolhido": int(escolhido),
                           "auc": round(a, 4), "achadas": round(r, 4)})
        txt.append("")

    # Furinho: além de achar o alvo, o candidato deve NÃO acusar os outros defeitos
    colunas = ["good"] + TIPOS
    txt += ["FURINHO: quantas cápsulas de cada tipo cada candidato acusa",
            "(acima do percentil 98 das boas de treino, 5 folds; o ideal é acusar só o furinho)",
            f"    {'':<60}" + "".join(
                f"{t[:7] + ' ' + str(int(((tipo == t) & ~cal).sum())):>12}" for t in colunas)]
    for d, chave, descricao, escolhido in CANDIDATOS:
        if d != "poke":
            continue
        acima = acima_cv(np.array([x[chave] for x in valores]), boas, cal, fold)
        txt.append(f"  {'*' if escolhido else ' '} {descricao:<60}" + "".join(
            f"{int((acima & (tipo == t) & ~cal).sum()):>12d}" for t in colunas))
    txt.append("")
    texto = "\n".join(txt)
    (DIR_RESULT / "tabela_candidatos.txt").write_text(texto, encoding="utf-8")
    with open(DIR_RESULT / "tabela_candidatos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tabela[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(tabela)
    print("\n" + texto)

    texto, linhas_m = tabela_morfologia(valores, tipo, boas, cal, fold)
    (DIR_RESULT / "tabela_morfologia.txt").write_text(texto, encoding="utf-8")
    with open(DIR_RESULT / "tabela_morfologia.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas_m[0].keys()), lineterminator="\n")
        w.writeheader()
        for l in linhas_m:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in l.items()})
    print("\n" + texto)
    print(f"Tabelas em {DIR_RESULT}")


if __name__ == "__main__":
    main()
