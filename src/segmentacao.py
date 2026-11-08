"""
Etapa 3 - Máscara final do defeito (golden template + morfologia completa).

Parte do golden template da Etapa 1 (pipeline.py) e acrescenta a limpeza
morfológica, medindo o efeito de cada operação sobre os falsos positivos:

    C0  |teste - ref| > T                            (só limiar, ROI = cápsula inteira)
    C1  + abertura 5x5                               (= Etapa 1 do Daniel)
    C2  + ROI erodida 9 px                           (tira o contorno da cápsula)
    C3  + fechamento 5x5                             (junta pedaços do mesmo defeito)
    C4  + preenchimento de buracos                   (manchas sólidas)
    C5  + remoção de componentes com área < A_MIN    (tira manchas pequenas)

Regras da partição (dados/particao.csv):
    - referências do golden template: conjunto 'banco' (30 normais);
    - T e A_MIN: calculados só com o conjunto 'calibracao' (20 normais);
    - avaliação: conjunto 'classificacao' (192 normais + 109 defeituosas),
      que nunca participa da escolha de parâmetros.

Saídas:
    resultados/etapa3/tabela_morfologia.txt|csv   efeito de cada operação
    resultados/etapa3/exemplo_*.png               passo a passo em exemplos
    dados/gerado/mascaras/<pasta>_<tipo>_<id>.png máscara final (C4), usada na Etapa 4
    dados/gerado/cache_golden.npz                 diferenças já calculadas (cache)

Uso:  python src/segmentacao.py          (~10 min na primeira vez; depois usa o cache)
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from particao import carregar_particao
from pipeline import (
    ROI_EROSAO, carregar_banco_referencias, carregar_imagem, carregar_mascara_gt,
    converter_cinza, filtro_bilateral, metricas_pixel, segmentar_capsula,
    selecionar_referencia,
)

RAIZ = Path(__file__).resolve().parent.parent   # raiz do repositório (este arquivo está em src/)
GERADO = RAIZ / "dados" / "gerado"
CACHE = GERADO / "cache_golden.npz"
DIR_MASCARAS = GERADO / "mascaras"
DIR_RESULT = RAIZ / "resultados" / "etapa3"

ABERTURA = 5
FECHAMENTO = 5
PERCENTIL_LIMIAR = 99   # T = percentil 99 da diferença entre cápsulas boas
PERCENTIL_AREA = 95     # A_MIN = percentil 95 da maior mancha que sobra em cápsulas boas
MIN_PX_HIT = 30         # mesmo critério de "defeito encontrado" da Etapa 1

# Máscara levada para a Etapa 4. C5 é medida, mas NÃO é usada: em cápsulas boas, as
# manchas falsas (impressão "500" deslocada) chegam a ser tão grandes quanto os
# defeitos, e A_MIN calibrado nas boas apaga quase todos os defeitos (ver tabela).
# A decisão "mancha falsa x defeito" fica para os atributos e o classificador.
ETAPA_FINAL = "C4 +preenchimento"

ETAPAS = ["C0 limiar", "C1 +abertura", "C2 +ROI erodida", "C3 +fechamento",
          "C4 +preenchimento", "C5 +remove pequenos"]


def _ee(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# =============================================================================
# 1. DIFERENÇA DO GOLDEN TEMPLATE (com cache)
# =============================================================================
def calcular_diferencas(linhas: list[dict], banco: list[dict]) -> dict[str, dict]:
    """Para cada imagem: |teste alinhado - referência mais parecida| e as ROIs.

    Guarda em cache (dados/gerado/cache_golden.npz), porque a escolha da referência
    alinha cada imagem às 30 do banco e é a parte lenta.
    """
    cache = {}
    if CACHE.exists():
        with np.load(CACHE) as z:
            cache = {k: z[k] for k in z.files}
    novos = 0
    saida = {}
    for i, l in enumerate(linhas):
        chave = l["arquivo"].replace("/", "__")
        if chave + "__diff" not in cache:
            img = filtro_bilateral(converter_cinza(carregar_imagem(l["caminho"])))
            ref, alinhada = selecionar_referencia(img, banco)
            cache[chave + "__diff"] = cv2.absdiff(alinhada, ref["imagem"])
            cache[chave + "__roi"] = ref["mascara"]
            novos += 1
            if novos % 25 == 0:
                print(f"  golden template: {i + 1}/{len(linhas)}")
        roi_cheia = cache[chave + "__roi"]
        saida[l["arquivo"]] = {
            "diff": cache[chave + "__diff"],
            "roi_cheia": roi_cheia,
            "roi_erodida": cv2.erode(roi_cheia, _ee(ROI_EROSAO)),
        }
    if novos:
        GERADO.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, **cache)
    return saida


# =============================================================================
# 2. MORFOLOGIA
# =============================================================================
def preencher_buracos(mascara: np.ndarray) -> np.ndarray:
    """Preenche buracos internos: pinta o fundo a partir da borda (flood fill) e
    tudo que não foi alcançado é buraco dentro de alguma mancha."""
    fundo = mascara.copy()
    h, w = mascara.shape
    borda = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(fundo, borda, (0, 0), 255)
    return mascara | cv2.bitwise_not(fundo)


def remover_pequenos(mascara: np.ndarray, area_min: int) -> np.ndarray:
    """Remove componentes conexos (8-vizinhança) com área menor que area_min."""
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    manter = np.zeros(n, bool)
    manter[1:] = stats[1:, cv2.CC_STAT_AREA] >= area_min
    return np.where(manter[rotulos], 255, 0).astype(np.uint8)


def etapas_mascara(d: dict, limiar: int, area_min: int | None) -> dict[str, np.ndarray]:
    """Aplica C0..C5 em sequência e devolve a máscara de cada etapa."""
    m = {}
    c0 = np.where(d["diff"] > limiar, 255, 0).astype(np.uint8) & d["roi_cheia"]
    m[ETAPAS[0]] = c0
    c1 = cv2.morphologyEx(c0, cv2.MORPH_OPEN, _ee(ABERTURA))
    m[ETAPAS[1]] = c1
    c2 = c1 & d["roi_erodida"]
    m[ETAPAS[2]] = c2
    c3 = cv2.morphologyEx(c2, cv2.MORPH_CLOSE, _ee(FECHAMENTO)) & d["roi_erodida"]
    m[ETAPAS[3]] = c3
    c4 = preencher_buracos(c3)
    m[ETAPAS[4]] = c4
    if area_min is not None:
        m[ETAPAS[5]] = remover_pequenos(c4, area_min)
    return m


# =============================================================================
# 3. CALIBRAÇÃO (só cápsulas boas do conjunto 'calibracao')
# =============================================================================
def calibrar(dif_cal: dict[str, dict]) -> tuple[int, int]:
    """T = percentil 99 da diferença dentro da cápsula; A_MIN = percentil 95 da
    maior mancha que sobra em C4 em cada cápsula boa (ou seja, com A_MIN, ~95%
    das cápsulas boas de calibração ficam sem nenhuma mancha)."""
    amostras = np.concatenate([d["diff"][d["roi_cheia"] > 0] for d in dif_cal.values()])
    limiar = int(np.ceil(np.percentile(amostras, PERCENTIL_LIMIAR)))
    maiores = []
    for d in dif_cal.values():
        c4 = etapas_mascara(d, limiar, None)[ETAPAS[4]]
        n, _, stats, _ = cv2.connectedComponentsWithStats(c4, connectivity=8)
        maiores.append(stats[1:, cv2.CC_STAT_AREA].max() if n > 1 else 0)
    area_min = int(np.ceil(np.percentile(maiores, PERCENTIL_AREA))) + 1
    return limiar, area_min


# =============================================================================
# 4. AVALIAÇÃO (conjunto 'classificacao')
# =============================================================================
def avaliar(linhas: list[dict], dif: dict[str, dict], limiar: int, area_min: int):
    tipos = ["crack", "faulty_imprint", "poke", "scratch", "squeeze"]
    reg = {e: [] for e in ETAPAS}
    finais = {}
    for l in linhas:
        d = dif[l["arquivo"]]
        gt = (np.zeros_like(d["diff"]) if l["tipo"] == "good"
              else carregar_mascara_gt(l["tipo"], int(Path(l["arquivo"]).stem), d["diff"].shape))
        masc = etapas_mascara(d, limiar, area_min)
        for e, m in masc.items():
            reg[e].append((l["tipo"], metricas_pixel(m, gt), bool(m.any())))
        finais[l["arquivo"]] = masc[ETAPA_FINAL]

    linhas_tab = []
    for e in ETAPAS:
        defe = [(t, m) for t, m, _ in reg[e] if t != "good"]
        boas = [(m, a) for t, m, a in reg[e] if t == "good"]
        tp = sum(m["tp"] for _, m in defe)
        fp = sum(m["fp"] for _, m in defe)
        linha = {
            "etapa": e,
            "hit": np.mean([m["tp"] >= MIN_PX_HIT for _, m in defe]),
            "FP_px_boa": np.mean([m["fp"] for m, _ in boas]),
            "boas_com_alarme": np.mean([a for _, a in boas]),
            "precisao": tp / (tp + fp + 1e-9),
            "recall": np.mean([m["recall"] for _, m in defe]),
            "dice": np.mean([m["dice"] for _, m in defe]),
        }
        for t in tipos:
            linha[f"dice_{t}"] = np.mean([m["dice"] for tt, m in defe if tt == t])
            linha[f"hit_{t}"] = np.mean([m["tp"] >= MIN_PX_HIT for tt, m in defe if tt == t])
        linhas_tab.append(linha)
    return linhas_tab, finais


def salvar_tabela(tab: list[dict], limiar: int, area_min: int) -> str:
    DIR_RESULT.mkdir(parents=True, exist_ok=True)
    with open(DIR_RESULT / "tabela_morfologia.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tab[0].keys()))
        w.writeheader()
        for l in tab:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in l.items()})

    txt = [f"Etapa 3 - efeito da morfologia (conjunto 'classificacao': 192 boas + 109 defeituosas)",
           f"Parâmetros calibrados só com as 20 boas de 'calibracao': T = {limiar}, A_MIN = {area_min} px",
           "",
           f"{'etapa':<22}{'hit':>6}{'FP px/boa':>11}{'boas c/ alarme':>16}"
           f"{'precisão':>10}{'recall':>8}{'dice':>7}"]
    for l in tab:
        txt.append(f"{l['etapa']:<22}{l['hit']:6.2f}{l['FP_px_boa']:11.0f}{l['boas_com_alarme']:16.0%}"
                   f"{l['precisao']:10.2f}{l['recall']:8.2f}{l['dice']:7.3f}")
    txt += ["", "Dice | hit por tipo de defeito:",
            f"{'etapa':<22}" + "".join(f"{t:>18}" for t in
                                       ["crack", "faulty_imprint", "poke", "scratch", "squeeze"])]
    for l in tab:
        txt.append(f"{l['etapa']:<22}" + "".join(
            f"{l['dice_' + t]:>11.2f} | {l['hit_' + t]:.2f}"
            for t in ["crack", "faulty_imprint", "poke", "scratch", "squeeze"]))
    texto = "\n".join(txt)
    (DIR_RESULT / "tabela_morfologia.txt").write_text(texto + "\n", encoding="utf-8")
    return texto


def figura_exemplos(linhas: list[dict], dif: dict, limiar: int, area_min: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exemplos = ["test/squeeze/001.png", "test/crack/000.png", "test/good/005.png"]
    exemplos = [e for e in exemplos if e in dif]
    for arq in exemplos:
        d = dif[arq]
        tipo = arq.split("/")[1]
        img = converter_cinza(carregar_imagem(Path(next(l["caminho"] for l in linhas
                                                         if l["arquivo"] == arq))))
        gt = (np.zeros_like(img) if tipo == "good"
              else carregar_mascara_gt(tipo, int(Path(arq).stem), img.shape))
        masc = etapas_mascara(d, limiar, area_min)
        fig, eixos = plt.subplots(2, 4, figsize=(18, 7.5))
        paineis = [(img, f"{arq}\n(verde = defeito real)", "gray"),
                   (d["diff"], "|teste - referência|", "hot")] + \
                  [(m, e, "gray") for e, m in masc.items()]
        for ax, (im, tit, cmap) in zip(eixos.ravel(), paineis):
            ax.imshow(im[150:850, 0:1000], cmap=cmap, vmax=80 if cmap == "hot" else None)
            if gt.any():
                ax.contour(gt[150:850, 0:1000] > 0, levels=[0.5], colors="lime", linewidths=1)
            if cmap == "gray" and tit in masc:
                mt = metricas_pixel(masc[tit], gt)
                tit += f"\nFP={mt['fp']} px" + (f" | Dice={mt['dice']:.2f}" if gt.any() else "")
            ax.set_title(tit, fontsize=10)
            ax.axis("off")
        fig.suptitle(f"Etapa 3 - passo a passo da máscara (T={limiar}, A_MIN={area_min} px)")
        fig.tight_layout()
        fig.savefig(DIR_RESULT / f"exemplo_{tipo}_{Path(arq).stem}.png", dpi=80)
        plt.close(fig)


# =============================================================================
def main() -> None:
    banco = carregar_banco_referencias([l["caminho"] for l in carregar_particao("banco")])
    cal = carregar_particao("calibracao")
    cls = carregar_particao("classificacao")

    print("Calculando diferenças (calibração + classificação)...")
    dif = calcular_diferencas(cal + cls, banco)

    limiar, area_min = calibrar({l["arquivo"]: dif[l["arquivo"]] for l in cal})
    print(f"Calibrado em 20 boas: T = {limiar}, A_MIN = {area_min} px\n")

    tab, finais = avaliar(cls, dif, limiar, area_min)
    print(salvar_tabela(tab, limiar, area_min))

    DIR_MASCARAS.mkdir(parents=True, exist_ok=True)
    for arq, m in finais.items():
        cv2.imwrite(str(DIR_MASCARAS / arq.replace("/", "_")), m)
    figura_exemplos(cls, dif, limiar, area_min)
    print(f"\nMáscaras finais em {DIR_MASCARAS} | tabelas e figuras em {DIR_RESULT}")


if __name__ == "__main__":
    main()
