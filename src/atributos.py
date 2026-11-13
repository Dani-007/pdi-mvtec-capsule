"""
Etapa 4 - Extração de atributos (a "ficha" de cada cápsula).

Entrada (por imagem do conjunto 'classificacao'):
    - máscara final C4 da Etapa 3 (dados/gerado/mascaras/), ou seja, as manchas suspeitas;
    - diferença do golden template (dados/gerado/cache_golden.npz);
    - imagem em cinza (bilateral) e máscara da cápsula (pipeline.py);
    - máscara da impressão "500" segmentada pelo histograma (impressao.py).

Saída: dados/atributos.csv, uma linha por imagem, com as colunas de identificação
(arquivo, tipo, rotulo, fold) e os atributos abaixo, organizados em 3 grupos para a
comparação pedida no enunciado (forma x textura x todos):

    FORMA (geometria da maior mancha + quantidade/área de manchas)
        n_manchas, area_total, area_maior, perimetro, compacidade, excentricidade,
        extensao, solidez, hu1, hu2, hu3
    TEXTURA (intensidade e textura dentro da maior mancha)
        media_cinza, desvio_cinza, media_dif, max_dif,
        suavidade, assimetria, uniformidade, entropia        (estatísticas do histograma)
        lbp_entropia, lbp_uniformidade                         (LBP 8 vizinhos, raio 1)
    POSIÇÃO E IMPRESSÃO (onde está a mancha e como está o "500")
        frac_sobre_500, frac_metade_preta, dist_borda,
        imp_area_rel, imp_componentes, imp_buracos, imp_cv_area_algarismos

Cápsula sem nenhuma mancha: atributos de forma/textura da mancha = 0 (n_manchas = 0 já
informa o classificador). Os atributos da impressão existem sempre.

A única referência usada para normalizar (área típica do "500") vem do conjunto
'calibracao'. Nada aqui olha os rótulos: eles só são copiados para o CSV.

Uso:  python src/atributos.py      (exige ter rodado src/segmentacao.py antes)
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from impressao import descrever_impressao, segmentar_impressao
from particao import carregar_particao
from pipeline import RAIZ, carregar_imagem, converter_cinza, filtro_bilateral, segmentar_capsula
from segmentacao import CACHE, DIR_MASCARAS

SAIDA = RAIZ / "dados" / "atributos.csv"
DIR_RESULT = RAIZ / "resultados" / "etapa4"
DILATA_500 = 15          # tolerância (px) ao redor da impressão
FUNDO_PRETO = 45         # fundo local (mediana 45x45) < 45 = metade preta

GRUPOS = {
    "forma": ["n_manchas", "area_total", "area_maior", "perimetro", "compacidade",
              "excentricidade", "extensao", "solidez", "hu1", "hu2", "hu3"],
    "textura": ["media_cinza", "desvio_cinza", "media_dif", "max_dif", "suavidade",
                "assimetria", "uniformidade", "entropia", "lbp_entropia", "lbp_uniformidade"],
    "posicao_impressao": ["frac_sobre_500", "frac_metade_preta", "dist_borda", "imp_area_rel",
                          "imp_componentes", "imp_buracos", "imp_cv_area_algarismos"],
}
ATRIBUTOS = [a for g in GRUPOS.values() for a in g]


def _ee(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# -----------------------------------------------------------------------------
# Descritores
# -----------------------------------------------------------------------------
def lbp(cinza: np.ndarray) -> np.ndarray:
    """LBP básico: compara cada pixel com os 8 vizinhos (raio 1) e forma um código de 8 bits."""
    c = cinza.astype(np.int16)
    p = np.pad(c, 1, mode="edge")
    h, w = c.shape
    vizinhos = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    codigo = np.zeros((h, w), np.uint8)
    for bit, (dy, dx) in enumerate(vizinhos):
        codigo |= ((p[dy:dy + h, dx:dx + w] >= c).astype(np.uint8) << bit)
    return codigo


def estatisticas_histograma(valores: np.ndarray, niveis: int = 256) -> dict[str, float]:
    """Descritores estatísticos de textura (momentos do histograma):
    suavidade R = 1 - 1/(1 + s^2), assimetria (3º momento), uniformidade e entropia."""
    hist = np.bincount(valores.ravel(), minlength=niveis).astype(float)
    p = hist / hist.sum()
    z = np.arange(niveis) / (niveis - 1)              # níveis normalizados em [0, 1]
    media = np.sum(z * p)
    var = np.sum((z - media) ** 2 * p)
    return {
        "suavidade": 1 - 1 / (1 + var),
        "assimetria": float(np.sum((z - media) ** 3 * p)),
        "uniformidade": float(np.sum(p ** 2)),
        "entropia": float(-np.sum(p[p > 0] * np.log2(p[p > 0]))),
    }


def descrever_forma(maior: np.ndarray) -> dict[str, float]:
    """Geometria da maior mancha (máscara binária com um único componente)."""
    contornos, _ = cv2.findContours(maior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(contornos, key=cv2.contourArea)
    area = float(np.sum(maior > 0))
    perim = cv2.arcLength(cont, True)
    m = cv2.moments(maior, binaryImage=True)
    # excentricidade pelos autovalores da matriz de covariância (momentos centrais)
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    d = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    l1, l2 = (mu20 + mu02) / 2 + d, (mu20 + mu02) / 2 - d
    exc = float(np.sqrt(1 - l2 / l1)) if l1 > 0 else 0.0
    _, _, w, h = cv2.boundingRect(cont)
    casco = cv2.contourArea(cv2.convexHull(cont))
    hu = cv2.HuMoments(m).ravel()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)  # escala logarítmica
    return {
        "area_maior": area,
        "perimetro": perim,
        "compacidade": perim ** 2 / (4 * np.pi * area),   # 1 = círculo; cresce com o alongamento
        "excentricidade": exc,
        "extensao": area / (w * h),
        "solidez": area / casco if casco > 0 else 1.0,
        "hu1": hu[0], "hu2": hu[1], "hu3": hu[2],
    }


def ficha(cinza: np.ndarray, capsula: np.ndarray, manchas: np.ndarray, dif: np.ndarray,
          imp: np.ndarray, area_500_ref: float, codigos_lbp: np.ndarray) -> dict[str, float]:
    f = {a: 0.0 for a in ATRIBUTOS}

    # Impressão (existe sempre)
    di = descrever_impressao(imp)
    f["imp_area_rel"] = di["imp_area"] / area_500_ref
    f["imp_componentes"] = di["imp_componentes"]
    f["imp_buracos"] = di["imp_buracos"]
    f["imp_cv_area_algarismos"] = di["imp_cv_area_algarismos"]

    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(manchas, connectivity=8)
    f["n_manchas"] = float(n - 1)
    if n <= 1:
        return f
    f["area_total"] = float(stats[1:, cv2.CC_STAT_AREA].sum())
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    maior = np.where(rotulos == i, 255, 0).astype(np.uint8)
    sel = maior > 0

    f.update(descrever_forma(maior))

    f["media_cinza"] = float(cinza[sel].mean())
    f["desvio_cinza"] = float(cinza[sel].std())
    f["media_dif"] = float(dif[sel].mean())
    f["max_dif"] = float(dif[sel].max())
    f.update(estatisticas_histograma(cinza[sel]))
    lb = estatisticas_histograma(codigos_lbp[sel])
    f["lbp_entropia"], f["lbp_uniformidade"] = lb["entropia"], lb["uniformidade"]

    zona_500 = cv2.dilate(imp, _ee(DILATA_500)) > 0
    f["frac_sobre_500"] = float(np.mean(zona_500[sel]))
    fundo = cv2.medianBlur(cinza, 45)
    f["frac_metade_preta"] = float(np.mean(fundo[sel] < FUNDO_PRETO))
    dist = cv2.distanceTransform(capsula, cv2.DIST_L2, 5)        # distância ao contorno
    f["dist_borda"] = float(np.mean(dist[sel]) / (dist.max() + 1e-9))
    return f


# -----------------------------------------------------------------------------
def preparar(caminho: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cinza = filtro_bilateral(converter_cinza(carregar_imagem(caminho)))
    capsula, _ = segmentar_capsula(cinza)
    imp, _ = segmentar_impressao(cinza, capsula)
    return cinza, capsula, imp


def area_500_calibracao() -> float:
    areas = [descrever_impressao(preparar(l["caminho"])[2])["imp_area"]
             for l in carregar_particao("calibracao")]
    return float(np.median(areas))


def figura(linhas: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tipos = ["good", "crack", "faulty_imprint", "poke", "scratch", "squeeze"]
    mostrar = ["n_manchas", "area_maior", "compacidade", "excentricidade", "media_dif",
               "entropia", "frac_sobre_500", "dist_borda", "imp_area_rel", "imp_buracos"]
    fig, eixos = plt.subplots(2, 5, figsize=(22, 8))
    for ax, a in zip(eixos.ravel(), mostrar):
        dados = [[l[a] for l in linhas if l["tipo"] == t] for t in tipos]
        ax.boxplot(dados, showfliers=False)
        ax.set_xticks(range(1, len(tipos) + 1))
        ax.set_xticklabels(["boa", "crack", "imprint", "poke", "scratch", "squeeze"],
                           rotation=30, fontsize=9)
        ax.set_title(a)
    fig.suptitle("Etapa 4 - distribuição de alguns atributos por tipo (conjunto classificação)")
    fig.tight_layout()
    fig.savefig(DIR_RESULT / "atributos_por_tipo.png", dpi=80)
    plt.close(fig)


def main() -> None:
    if not CACHE.exists() or not DIR_MASCARAS.exists():
        raise SystemExit("Rode antes:  python src/segmentacao.py")
    ref_500 = area_500_calibracao()
    print(f"Área típica do '500' (mediana nas 20 boas de calibração): {ref_500:.0f} px")

    with np.load(CACHE) as z:
        cache = {k: z[k] for k in z.files}
    saida = []
    cls = carregar_particao("classificacao")
    for k, l in enumerate(cls, 1):
        cinza, capsula, imp = preparar(l["caminho"])
        manchas = cv2.imread(str(DIR_MASCARAS / l["arquivo"].replace("/", "_")),
                             cv2.IMREAD_GRAYSCALE)
        dif = cache[l["arquivo"].replace("/", "__") + "__diff"]
        f = ficha(cinza, capsula, manchas, dif, imp, ref_500, lbp(cinza))
        saida.append({"arquivo": l["arquivo"], "tipo": l["tipo"], "rotulo": l["rotulo"],
                      "fold": l["fold"], **f})
        if k % 50 == 0:
            print(f"  {k}/{len(cls)}")

    with open(SAIDA, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["arquivo", "tipo", "rotulo", "fold"] + ATRIBUTOS,
                           lineterminator="\n")
        w.writeheader()
        for l in saida:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in l.items()})
    DIR_RESULT.mkdir(parents=True, exist_ok=True)
    figura(saida)
    print(f"{len(saida)} fichas com {len(ATRIBUTOS)} atributos em {SAIDA}")
    print(f"Figura em {DIR_RESULT / 'atributos_por_tipo.png'}")


if __name__ == "__main__":
    main()
