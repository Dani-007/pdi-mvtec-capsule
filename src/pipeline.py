"""Etapa 1: funções básicas do pipeline e localização do defeito (bordas x golden template).

Uso:
    python src/pipeline.py --defeito crack --indice 0     # passo a passo de uma imagem
    python src/pipeline.py --calibrar                     # recalibra o limiar do golden template

Convenções: imagens em cinza uint8; máscaras uint8 com valores {0, 255}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# --- PARÂMETROS ---
# Cinza após o bilateral: fundo ~200, laranja ~85-95, preto ~20-40, "500" ~86-121.
RAIZ = Path(__file__).resolve().parent.parent
DATASET_DIR = RAIZ / "capsule"

# Bilateral: o ruído varia ~5-15 níveis; sigma_cor 40 o suaviza e preserva saltos grandes.
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 40
BILATERAL_SIGMA_SPACE = 9

# Canny: alto 40 inicia ~1-2% dos pixels da cápsula (p99 do gradiente ~60); baixo 15 continua
# bordas fracas (histerese). L2gradient: magnitude euclidiana, melhor em bordas diagonais.
CANNY_LIMIAR_BAIXO = 15
CANNY_LIMIAR_ALTO = 40
CANNY_APERTURE = 3

# Sobel 3x3, limiar fixo ~p99 da magnitude dentro da cápsula.
SOBEL_KSIZE = 3
SOBEL_LIMIAR = 60

# Abertura com elipse de k px apaga o que for mais estreito que k. O Canny dá bordas de 1 px:
# um fechamento de 5 px as agrupa antes, senão a abertura apagaria todas.
MORF_TAMANHOS = (3, 5, 7, 9)
MORF_FECHAMENTO_PREVIO = 5

# Erosão da máscara da cápsula: tira o contorno, que sempre acende (~93% do GT continua dentro).
ROI_EROSAO = 9

# Golden template: limiar ~ p99 da diferença entre boas; abertura 5 tira as lascas de 1-3 px
# do erro de alinhamento, mais finas que os defeitos.
GOLDEN_LIMIAR = 30
GOLDEN_ABERTURA = 5
GOLDEN_N_BANCO = 30        # cápsulas boas candidatas a referência
ECC_ESCALA = 0.25          # alinhamento em 1/4 da resolução: rápido e menos sensível ao ruído
ECC_ITERACOES = 200
ECC_EPS = 1e-6


# --- E/S ---
def carregar_imagem(caminho: str | Path) -> np.ndarray:
    """Lê uma imagem BGR. Lança erro explícito se o arquivo não existir."""
    img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler: {caminho}")
    return img


def carregar_mascara_gt(defeito: str, indice: int, shape: tuple[int, int]) -> np.ndarray:
    """Máscara ground truth do MVTec (ou máscara vazia para a classe 'good')."""
    caminho = DATASET_DIR / "ground_truth" / defeito / f"{indice:03d}_mask.png"
    if defeito == "good" or not caminho.exists():
        return np.zeros(shape, np.uint8)
    gt = cv2.imread(str(caminho), cv2.IMREAD_GRAYSCALE)
    return np.where(gt > 127, 255, 0).astype(np.uint8)


def caminho_teste(defeito: str, indice: int) -> Path:
    return DATASET_DIR / "test" / defeito / f"{indice:03d}.png"


# --- 1. PRÉ-PROCESSAMENTO ---
def converter_cinza(img_bgr: np.ndarray) -> np.ndarray:
    """BGR -> tons de cinza (luminância ITU-R BT.601)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def filtro_bilateral(cinza: np.ndarray,
                     d: int = BILATERAL_D,
                     sigma_color: float = BILATERAL_SIGMA_COLOR,
                     sigma_space: float = BILATERAL_SIGMA_SPACE) -> np.ndarray:
    """Suaviza a superfície preservando bordas (peso espacial x peso de intensidade)."""
    return cv2.bilateralFilter(cinza, d, sigma_color, sigma_space)


def segmentar_capsula(cinza: np.ndarray, erosao: int = ROI_EROSAO) -> tuple[np.ndarray, np.ndarray]:
    """Máscara do objeto (cápsula) por Otsu + maior componente + preenchimento."""
    blur = cv2.GaussianBlur(cinza, (5, 5), 0)
    _, binaria = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mascara = np.zeros_like(cinza)
    if contornos:
        maior = max(contornos, key=cv2.contourArea)
        cv2.drawContours(mascara, [maior], -1, 255, thickness=cv2.FILLED)

    roi = mascara
    if erosao > 0:
        roi = cv2.erode(mascara, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosao, erosao)))
    return mascara, roi


def mascara_especular(img_bgr: np.ndarray, v_min: int = 230, s_max: int = 40) -> np.ndarray:
    """(Opcional/diagnóstico) Pixels de brilho especular: V alto e S baixo em HSV."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return np.where((hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max), 255, 0).astype(np.uint8)


# --- 2. DETECÇÃO DE BORDAS ---
def detectar_bordas_canny(img: np.ndarray,
                          limiar_baixo: int = CANNY_LIMIAR_BAIXO,
                          limiar_alto: int = CANNY_LIMIAR_ALTO,
                          aperture: int = CANNY_APERTURE,
                          roi: np.ndarray | None = None) -> np.ndarray:
    """Canny com histerese. Se `roi` for dada, descarta bordas fora dela."""
    bordas = cv2.Canny(img, limiar_baixo, limiar_alto,
                       apertureSize=aperture, L2gradient=True)
    if roi is not None:
        bordas = cv2.bitwise_and(bordas, roi)
    return bordas


def detectar_bordas_sobel(img: np.ndarray,
                          ksize: int = SOBEL_KSIZE,
                          limiar: float = SOBEL_LIMIAR,
                          roi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude do gradiente de Sobel + limiar fixo."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)

    bordas = mag > limiar
    if roi is not None:
        bordas &= roi > 0
    mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag_vis, np.where(bordas, 255, 0).astype(np.uint8)


# --- 3. MORFOLOGIA MATEMÁTICA ---
def aplicar_abertura(binaria: np.ndarray,
                     tamanhos: tuple[int, ...] = MORF_TAMANHOS,
                     forma: int = cv2.MORPH_ELLIPSE,
                     fechamento_previo: int = 0) -> dict[int, np.ndarray]:
    """Abertura (erosão seguida de dilatação) para cada tamanho de elemento estruturante."""
    base = binaria
    if fechamento_previo > 0:
        ef = cv2.getStructuringElement(forma, (fechamento_previo, fechamento_previo))
        base = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, ef)

    resultados = {}
    for k in tamanhos:
        ee = cv2.getStructuringElement(forma, (k, k))
        resultados[k] = cv2.morphologyEx(base, cv2.MORPH_OPEN, ee)
    return resultados


# --- 4. SEGMENTAÇÃO POR GOLDEN TEMPLATE ---
def alinhar_imagem(teste: np.ndarray, referencia: np.ndarray,
                   escala: float = ECC_ESCALA) -> tuple[np.ndarray, np.ndarray]:
    """Alinha `teste` à `referencia` (cinza) por ECC com modelo euclidiano (rotação +
    translação)."""
    pequeno_t = cv2.resize(teste, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
    pequeno_r = cv2.resize(referencia, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    warp = np.eye(2, 3, dtype=np.float32)
    criterio = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERACOES, ECC_EPS)
    try:
        _, warp = cv2.findTransformECC(pequeno_r.astype(np.float32), pequeno_t.astype(np.float32),
                                       warp, cv2.MOTION_EUCLIDEAN, criterio, None, 5)
    except cv2.error:
        print("[aviso] ECC não convergiu; usando imagem sem alinhamento.")
        return teste.copy(), np.eye(2, 3, dtype=np.float32)

    warp[:, 2] /= escala
    h, w = referencia.shape[:2]
    alinhada = cv2.warpAffine(teste, warp, (w, h),
                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
    return alinhada, warp


def carregar_banco_referencias(caminhos: list[Path]) -> list[dict]:
    """Pré-processa (cinza + bilateral) as cápsulas boas candidatas a golden template."""
    banco = []
    for c in caminhos:
        img = filtro_bilateral(converter_cinza(carregar_imagem(c)))
        mascara, _ = segmentar_capsula(img, erosao=0)
        banco.append({"imagem": img, "mascara": mascara, "caminho": c})
    return banco


def selecionar_referencia(teste: np.ndarray, banco: list[dict]) -> tuple[dict, np.ndarray]:
    """Escolhe a referência do banco mais parecida com a imagem de teste."""
    melhor, melhor_alinhada, melhor_custo = None, None, np.inf
    for ref in banco:
        alinhada, _ = alinhar_imagem(teste, ref["imagem"])
        custo = cv2.absdiff(alinhada, ref["imagem"])[ref["mascara"] > 0].mean()
        if custo < melhor_custo:
            melhor, melhor_alinhada, melhor_custo = ref, alinhada, custo
    return melhor, melhor_alinhada


def golden_template(teste_alinhado: np.ndarray, referencia: np.ndarray,
                    limiar: int = GOLDEN_LIMIAR,
                    abertura: int = GOLDEN_ABERTURA,
                    roi: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """|teste - referência| -> limiar -> abertura -> restrição à ROI."""
    diff = cv2.absdiff(teste_alinhado, referencia)
    _, mascara = cv2.threshold(diff, limiar, 255, cv2.THRESH_BINARY)
    if abertura > 0:
        ee = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abertura, abertura))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, ee)
    if roi is not None:
        mascara = cv2.bitwise_and(mascara, roi)
    return diff, mascara


def calibrar_limiar_golden(banco: list[dict], caminhos_bons: list[Path],
                           percentil: float = 99.0) -> int:
    """Escolhe o limiar do golden template a partir de imagens SEM defeito."""
    amostras = []
    for c in caminhos_bons:
        img = filtro_bilateral(converter_cinza(carregar_imagem(c)))
        ref, alinhada = selecionar_referencia(img, banco)
        amostras.append(cv2.absdiff(alinhada, ref["imagem"])[ref["mascara"] > 0])
    return int(np.ceil(np.percentile(np.concatenate(amostras), percentil)))


# --- MÉTRICAS PIXEL A PIXEL ---
def metricas_pixel(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Precisão, recall, Dice, IoU e contagem de falsos positivos (em pixels)."""
    p, g = pred > 0, gt > 0
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    eps = 1e-9
    return {
        "precisao": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "iou": tp / (tp + fp + fn + eps),
        "fp": fp,
        "tp": tp,
    }


# --- VISUALIZAÇÃO ---
def _sobrepor_gt(img_bgr: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Desenha o contorno do GT em verde sobre a imagem (RGB para o matplotlib)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).copy()
    contornos, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgb, contornos, -1, (0, 255, 0), 3)
    return rgb


def _titulo_metricas(nome: str, pred: np.ndarray, gt: np.ndarray) -> str:
    m = metricas_pixel(pred, gt)
    if np.any(gt):
        return f"{nome}\nFP={m['fp']} px | prec={m['precisao']:.2f}"
    return f"{nome}\nFP={m['fp']} px"


def visualizar_pipeline(etapas: dict, gt: np.ndarray, titulo: str = "",
                        salvar: str | Path | None = None) -> None:
    """Grid do passo a passo."""
    aberturas_c = etapas["abertura_canny"]
    aberturas_s = etapas["abertura_sobel"]
    n_col = 1 + max(len(aberturas_c), 3)

    fig, eixos = plt.subplots(3, n_col, figsize=(3.2 * n_col, 10))
    for ax in eixos.ravel():
        ax.axis("off")

    linha1 = [
        (_sobrepor_gt(etapas["original"], gt), "Original (GT em verde)", None),
        (etapas["cinza"], "Tons de cinza", "gray"),
        (etapas["bilateral"], "Bilateral", "gray"),
        (etapas["roi"], f"ROI (erosão {ROI_EROSAO}px)", "gray"),
    ]
    for ax, (img, nome, cmap) in zip(eixos[0], linha1):
        ax.imshow(img, cmap=cmap)
        ax.set_title(nome, fontsize=9)

    for i, (nome_det, bordas, aberturas) in enumerate([
        (f"Canny ({CANNY_LIMIAR_BAIXO}/{CANNY_LIMIAR_ALTO})", etapas["canny"], aberturas_c),
        (f"Sobel (|G| > {SOBEL_LIMIAR})", etapas["sobel"], aberturas_s),
    ], start=1):
        eixos[i, 0].imshow(bordas, cmap="gray")
        eixos[i, 0].set_title(_titulo_metricas(nome_det, bordas, gt), fontsize=9)
        for j, (k, m) in enumerate(aberturas.items(), start=1):
            eixos[i, j].imshow(m, cmap="gray")
            eixos[i, j].set_title(_titulo_metricas(f"Abertura {k}x{k}", m, gt), fontsize=9)

    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    if salvar:
        fig.savefig(salvar, dpi=110, bbox_inches="tight")
    plt.show()


def visualizar_golden(teste_bgr: np.ndarray, referencia: np.ndarray, alinhada: np.ndarray,
                      diff: np.ndarray, mascara: np.ndarray, gt: np.ndarray,
                      titulo: str = "", salvar: str | Path | None = None) -> None:
    """Grid do golden template: teste, referência, alinhada, |diferença|, máscara."""
    fig, eixos = plt.subplots(1, 5, figsize=(17, 4))
    paineis = [
        (_sobrepor_gt(teste_bgr, gt), "Teste (GT em verde)", None),
        (referencia, "Referência (golden)", "gray"),
        (alinhada, "Teste alinhado (ECC)", "gray"),
        (diff, "|teste - ref|", "inferno"),
        (mascara, _titulo_metricas("Máscara de anomalia", mascara, gt), "gray"),
    ]
    for ax, (img, nome, cmap) in zip(eixos, paineis):
        ax.imshow(img, cmap=cmap)
        ax.set_title(nome, fontsize=9)
        ax.axis("off")
    if np.any(gt):
        m = metricas_pixel(mascara, gt)
        titulo += f"   |   Dice={m['dice']:.3f}  IoU={m['iou']:.3f}  recall={m['recall']:.3f}"
    fig.suptitle(titulo, fontsize=12)
    fig.tight_layout()
    if salvar:
        fig.savefig(salvar, dpi=110, bbox_inches="tight")
    plt.show()


# --- ORQUESTRAÇÃO ---
def executar_pipeline_bordas(img_bgr: np.ndarray) -> dict:
    """Executa Denoising -> Bordas (Canny e Sobel) -> Abertura e devolve todas as etapas."""
    cinza = converter_cinza(img_bgr)
    suave = filtro_bilateral(cinza)
    _, roi = segmentar_capsula(suave)

    canny = detectar_bordas_canny(suave, roi=roi)
    mag_sobel, sobel = detectar_bordas_sobel(suave, roi=roi)

    return {
        "original": img_bgr,
        "cinza": cinza,
        "bilateral": suave,
        "roi": roi,
        "canny": canny,
        "sobel": sobel,
        "sobel_magnitude": mag_sobel,
        "abertura_canny": aplicar_abertura(canny, fechamento_previo=MORF_FECHAMENTO_PREVIO),
        "abertura_sobel": aplicar_abertura(sobel),
    }


def executar_golden(img_bgr: np.ndarray, banco: list[dict],
                    limiar: int = GOLDEN_LIMIAR) -> dict:
    """Executa Denoising -> Seleção de referência + Alinhamento -> Diferença -> Limiar/Abertura."""
    suave = filtro_bilateral(converter_cinza(img_bgr))
    ref, alinhada = selecionar_referencia(suave, banco)
    diff, mascara = golden_template(alinhada, ref["imagem"], limiar=limiar, roi=ref["mascara"])
    return {"referencia": ref["imagem"], "referencia_caminho": ref["caminho"],
            "alinhada": alinhada, "diff": diff, "mascara": mascara}


def main() -> None:
    parser = argparse.ArgumentParser(description="Localização de defeitos (capsule)")
    parser.add_argument("--defeito", default="scratch",
                        choices=["good", "crack", "faulty_imprint", "poke", "scratch", "squeeze"])
    parser.add_argument("--indice", type=int, default=0)
    parser.add_argument("--n-ref", type=int, default=GOLDEN_N_BANCO,
                        help="nº de imagens train/good no banco de referências")
    parser.add_argument("--calibrar", action="store_true",
                        help="calibra o limiar do golden template em outras imagens train/good")
    parser.add_argument("--salvar", default=None, help="prefixo para salvar as figuras (.png)")
    args = parser.parse_args()

    img = carregar_imagem(caminho_teste(args.defeito, args.indice))
    gt = carregar_mascara_gt(args.defeito, args.indice, img.shape[:2])
    titulo = f"capsule / {args.defeito} / {args.indice:03d}"

    # --- Bordas + morfologia -------------------------------------------------
    etapas = executar_pipeline_bordas(img)
    visualizar_pipeline(etapas, gt, titulo,
                        salvar=f"{args.salvar}_bordas.png" if args.salvar else None)

    # --- Golden template -----------------------------------------------------
    bons = sorted((DATASET_DIR / "train" / "good").glob("*.png"))
    banco = carregar_banco_referencias(bons[:args.n_ref])

    limiar = GOLDEN_LIMIAR
    if args.calibrar:
        limiar = calibrar_limiar_golden(banco, bons[args.n_ref:args.n_ref + 20])
        print(f"Limiar calibrado (p99 em 20 cápsulas boas fora do banco): {limiar}")

    g = executar_golden(img, banco, limiar=limiar)
    visualizar_golden(img, g["referencia"], g["alinhada"], g["diff"], g["mascara"], gt,
                      titulo=f"{titulo}  (ref={g['referencia_caminho'].name}, limiar={limiar})",
                      salvar=f"{args.salvar}_golden.png" if args.salvar else None)


if __name__ == "__main__":
    main()
