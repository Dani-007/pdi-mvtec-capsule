"""
Segmentação da impressão "500" pelo histograma dentro da cápsula (sugestão do professor).

Dentro da máscara da cápsula (sem o fundo), o histograma de intensidade tem modos separados:
    < 45     metade preta
    45-75    logo "actavis" (sobre a metade preta)
    75-112   metade laranja
    > 112    impressão "500" (a região mais clara da cápsula)

Um limiar fixo ou um Otsu direto nessa faixa clara falha em parte das imagens: a iluminação
deixa a metade laranja com DOIS tons (topo e base) e o Otsu corta no meio do laranja.
Por isso o histograma é calculado sobre o realce local, e não sobre o cinza:

    1. fundo local = filtro da mediana 45x45 (maior que a espessura dos algarismos, ~25 px);
    2. realce = cinza - fundo local  (quanto o pixel é mais claro que a vizinhança);
    3. região = máscara da cápsula erodida 21 px (tira o contorno, que também é claro)
                e só a metade clara (fundo local > 60), onde fica o "500";
    4. limiar de Otsu no histograma do realce dentro dessa região;
    5. abertura 5x5 e componentes conexos: ficam só os "algarismos" (área >= 300 px,
       altura >= 40 px e largura/altura < 3), o que descarta reflexos compridos.

O logo "actavis" (na metade preta) é segmentado do mesmo jeito, no histograma do realce
dentro da metade escura (segmentar_logo). Quando a cápsula está girada e o logo não aparece, o
limiar de Otsu cai para quase zero; por isso há um contraste mínimo (REALCE_MIN_LOGO).

Uso das máscaras:
  1. Na localização (segmentacao.py): a cápsula gira em torno do próprio eixo, então o "500"
     e o logo aparecem alguns pixels deslocados em relação à referência, mesmo depois do
     alinhamento global. Na região deles, a referência é realinhada localmente por template
     matching (realinhar_local) antes da subtração. A variação normal da impressão some e uma
     falha de impressão continua aparecendo (a forma é diferente, não só a posição).
  2. Nos atributos (atributos.py): o "500" NÃO é apagado da mancha, porque os defeitos também
     caem sobre ele (impressão falha ~89% da área, furinho e arranhão ~45%); ele vira atributos.

Uso (figura de diagnóstico):  python src/impressao.py
"""

from __future__ import annotations

import cv2
import numpy as np

EROSAO_ROI = 21
JANELA_FUNDO = 45        # mediana: maior que a espessura do traço dos algarismos
FUNDO_MIN_CLARO = 60     # fundo local > 60 = metade laranja
AREA_MIN = 300
ALTURA_MIN = 40
RAZAO_MAX = 3.0
FUNDO_MAX_ESCURO = 60    # fundo local <= 60 = metade preta (onde fica o logo)
REALCE_MIN_LOGO = 6      # sem esse contraste mínimo, considera que o logo não está visível
BUSCA_LOCAL = 20         # deslocamento máximo (px) procurado no realinhamento local
DILATA_ZONA = 15         # zona ao redor da impressão/logo onde a diferença é recalculada


def _ee(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def segmentar_impressao(cinza: np.ndarray, mascara_capsula: np.ndarray) -> tuple[np.ndarray, int]:
    """Máscara binária do "500" e o limiar (no realce) usado. Vazia se nada for encontrado."""
    roi = cv2.erode(mascara_capsula, _ee(EROSAO_ROI))
    fundo = cv2.medianBlur(cinza, JANELA_FUNDO)
    realce = cv2.subtract(cinza, fundo)
    regiao = (roi > 0) & (fundo > FUNDO_MIN_CLARO)
    valores = realce[regiao]
    if valores.size < 100:
        return np.zeros_like(cinza), 255
    t, _ = cv2.threshold(valores.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaria = np.where((realce > t) & regiao, 255, 0).astype(np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _ee(5))

    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    saida = np.zeros_like(binaria)
    for i in range(1, n):
        _, _, w, h, area = stats[i]
        if area >= AREA_MIN and h >= ALTURA_MIN and w / h < RAZAO_MAX:
            saida[rotulos == i] = 255
    return saida, int(t)


def segmentar_logo(cinza: np.ndarray, mascara_capsula: np.ndarray) -> np.ndarray:
    """Máscara do logo "actavis": histograma do realce local na metade preta, Otsu, abertura
    3x3 e componentes do tamanho de letras. Vazia se o logo não estiver visível."""
    roi = cv2.erode(mascara_capsula, _ee(EROSAO_ROI))
    fundo = cv2.medianBlur(cinza, JANELA_FUNDO)
    realce = cv2.subtract(cinza, fundo)
    regiao = (roi > 0) & (fundo <= FUNDO_MAX_ESCURO)
    valores = realce[regiao]
    if valores.size < 100:
        return np.zeros_like(cinza)
    t, _ = cv2.threshold(valores.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if t < REALCE_MIN_LOGO:
        return np.zeros_like(cinza)
    binaria = np.where((realce > t) & regiao, 255, 0).astype(np.uint8)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _ee(3))
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    saida = np.zeros_like(binaria)
    for i in range(1, n):
        _, _, w, h, area = stats[i]
        if area >= 60 and w / h < 6 and h / w < 6:
            saida[rotulos == i] = 255
    return saida


def realinhar_local(dif: np.ndarray, teste: np.ndarray, ref: np.ndarray,
                    zona: np.ndarray) -> np.ndarray:
    """Recalcula |teste - ref| dentro da zona (impressão ou logo), depois de achar, por
    template matching, o pequeno deslocamento que melhor encaixa a referência no teste."""
    if not zona.any():
        return dif
    ys, xs = np.where(zona > 0)
    h, w = teste.shape
    b = BUSCA_LOCAL
    y0, y1 = max(ys.min() - 10, b), min(ys.max() + 10, h - b - 1)
    x0, x1 = max(xs.min() - 10, b), min(xs.max() + 10, w - b - 1)
    if y1 <= y0 or x1 <= x0:
        return dif
    recorte = teste[y0:y1, x0:x1]
    vizinhanca = ref[y0 - b:y1 + b, x0 - b:x1 + b]
    custo = cv2.matchTemplate(vizinhanca, recorte, cv2.TM_SQDIFF_NORMED)
    _, _, (mx, my), _ = cv2.minMaxLoc(custo)
    dx, dy = mx - b, my - b
    local = cv2.absdiff(recorte, ref[y0 + dy:y1 + dy, x0 + dx:x1 + dx])
    dentro = cv2.dilate(zona, _ee(DILATA_ZONA))[y0:y1, x0:x1] > 0
    saida = dif.copy()
    saida[y0:y1, x0:x1][dentro] = local[dentro]
    return saida


def diferenca_com_impressao(teste: np.ndarray, ref: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """|teste - ref| com o "500" e o logo realinhados localmente (ver docstring do módulo).
    `teste` já está alinhado globalmente à referência; `mascara` é a máscara da referência."""
    dif = cv2.absdiff(teste, ref)
    zona_500 = segmentar_impressao(teste, mascara)[0] | segmentar_impressao(ref, mascara)[0]
    dif = realinhar_local(dif, teste, ref, zona_500)
    zona_logo = segmentar_logo(teste, mascara) | segmentar_logo(ref, mascara)
    return realinhar_local(dif, teste, ref, zona_logo)


def descrever_impressao(mascara_500: np.ndarray) -> dict[str, float]:
    """Atributos de forma e topologia da impressão.

    Um "500" normal tem ~3-4 componentes (o traço de cima do "5" às vezes se separa) e 2
    buracos (os dois zeros). Uma falha de impressão muda a área, quebra algarismos
    (mais componentes) ou abre os zeros (menos buracos).
    """
    n, _, stats, _ = cv2.connectedComponentsWithStats(mascara_500, connectivity=8)
    componentes = n - 1
    contornos, hier = cv2.findContours(mascara_500, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    buracos = 0 if hier is None else int(np.sum(hier[0][:, 3] >= 0))
    area = float(np.sum(mascara_500 > 0))
    if componentes:
        areas = stats[1:, cv2.CC_STAT_AREA].astype(float)
        cv_areas = float(areas.std() / areas.mean()) if componentes > 1 else 0.0
    else:
        cv_areas = 0.0
    return {
        "imp_area": area,
        "imp_componentes": float(componentes),
        "imp_buracos": float(buracos),
        "imp_euler": float(componentes - buracos),
        "imp_cv_area_algarismos": cv_areas,   # variação de tamanho entre os algarismos
    }


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from pipeline import DATASET_DIR, RAIZ, carregar_imagem, converter_cinza, filtro_bilateral, \
        segmentar_capsula

    exemplos = ["train/good/051.png", "test/good/005.png", "train/good/120.png",
                "test/faulty_imprint/000.png", "test/faulty_imprint/005.png"]
    fig, eixos = plt.subplots(2, len(exemplos), figsize=(5 * len(exemplos), 6.5))
    for col, arq in enumerate(exemplos):
        g = filtro_bilateral(converter_cinza(carregar_imagem(DATASET_DIR / arq)))
        m, _ = segmentar_capsula(g)
        p, t = segmentar_impressao(g, m)
        d = descrever_impressao(p)
        eixos[0, col].imshow(g[250:750, 50:950], cmap="gray")
        eixos[0, col].set_title(arq)
        eixos[1, col].imshow(p[250:750, 50:950], cmap="gray")
        eixos[1, col].set_title(f"limiar(realce)={t} | área={d['imp_area']:.0f}\n"
                                f"componentes={d['imp_componentes']:.0f} | "
                                f"buracos={d['imp_buracos']:.0f} | Euler={d['imp_euler']:.0f}")
    for ax in eixos.ravel():
        ax.axis("off")
    fig.suptitle('Impressão "500" segmentada pelo histograma dentro da cápsula')
    fig.tight_layout()
    saida = RAIZ / "resultados" / "etapa4" / "impressao_500.png"
    saida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(saida, dpi=80)
    print(f"Figura salva em {saida}")


if __name__ == "__main__":
    main()
