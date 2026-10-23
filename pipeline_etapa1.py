"""
Projeto 4 (PDI - UFSCar) - Etapa 1: Localização do objeto/defeito
Categoria MVTec AD: capsule

Pipeline clássico (sem Deep Learning):
    1. Denoising ........ filtro bilateral
    2. Bordas ........... Canny  |  Sobel (magnitude + limiar fixo)
    3. Morfologia ....... abertura (MORPH_OPEN) com vários tamanhos de elemento estruturante
    4. Segmentação ...... golden template (diferença absoluta contra a referência boa
                          mais parecida, alinhada por ECC)

Uso:
    python pipeline_etapa1.py                                  # exemplo padrão (scratch/000)
    python pipeline_etapa1.py --defeito crack --indice 0
    python pipeline_etapa1.py --defeito good  --indice 3 --salvar saida.png

Convenções:
    - Imagens BGR (padrão OpenCV) só na leitura; o resto do pipeline trabalha em tons de cinza uint8.
    - Máscaras binárias são uint8 com valores {0, 255}.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# PARÂMETROS (centralizados para facilitar o estudo de sensibilidade)
# =============================================================================
# As imagens de capsule têm 1000x1000 px e a cápsula ocupa ~700x230 px.
# Contraste medido em tons de cinza (após bilateral):
#   - fundo ~200 | metade laranja ~85-95 | metade preta ~20-40
#   - texto "500" sobre o laranja: ~86 -> ~121 (só ~35 níveis), transição de 6-8 px
#   - arranhões na metade laranja: poucos níveis de cinza (defeito mais difícil)
# Os números entre colchetes nos comentários vêm de avaliar_etapa1.py, rodado nas
# 132 imagens de test/ (109 defeituosas + 23 boas).

DATASET_DIR = Path(__file__).resolve().parent / "capsule"

# --- Filtro bilateral ---------------------------------------------------------
# d = 9: vizinhança de 9x9 px. Cobre o grão do ruído do sensor e a textura fina
#   da gelatina; é bem menor que a extensão dos defeitos (dezenas de px).
#   d > 9 deixa o filtro lento (custo ~ d²) sem ganho visível em 1000x1000.
# sigmaColor = 40: o ruído na superfície varia ~5-15 níveis; bordas de texto,
#   rachaduras e "poke" variam >= 30. Com sigma=40, flutuações pequenas recebem
#   peso alto (são suavizadas) e saltos grandes recebem peso baixo (preservados).
#   O perfil de intensidade através do "500" fica praticamente igual para
#   sigmaColor entre 10 e 75, ou seja, a escolha é estável nessa faixa.
# sigmaSpace = 9: coerente com d=9 (quando d > 0 o OpenCV usa d como diâmetro).
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 40
BILATERAL_SIGMA_SPACE = 9

# --- Canny --------------------------------------------------------------------
# Razão baixo:alto = 1:2.7, dentro da faixa 1:2-1:3 recomendada por Canny.
# As magnitudes estão na escala do Sobel 3x3 não normalizado (0 a ~1000); dentro
# da cápsula, p50=4, p90=21 e p99=60.
# alto = 40: só ~1-2% dos pixels da cápsula iniciam bordas. Com 60 a impressão
#   "500" quase some, mas metade dos defeitos também [hit 0.50 vs 0.72 com 40].
# baixo = 15: a histerese continua uma borda forte por trechos fracos (arranhões
#   que perdem contraste). Com 10/25 a detecção sobe [hit 0.89], mas cada imagem
#   boa passa a ter ~2600 px de borda espúria (reflexos e textura).
# L2gradient=True: magnitude euclidiana, mais precisa que |gx|+|gy| em bordas
#   diagonais (arranhões raramente são alinhados aos eixos).
CANNY_LIMIAR_BAIXO = 15
CANNY_LIMIAR_ALTO = 40
CANNY_APERTURE = 3

# --- Sobel --------------------------------------------------------------------
# ksize = 3: menor suporte e melhor localização; ksize=5 engorda as bordas e
#   funde traços vizinhos das letras.
# Limiar fixo na magnitude bruta (mesma escala do Canny). O Otsu foi descartado:
#   dentro da cápsula ele separa "liso" de "qualquer gradiente" e marcava ~20 mil
#   px por imagem boa. limiar=60 (~p99 da cápsula) [hit 0.38, ~2600 FP px/boa].
SOBEL_KSIZE = 3
SOBEL_LIMIAR = 60

# --- Morfologia ---------------------------------------------------------------
# Elemento elíptico: isotrópico, não privilegia bordas horizontais/verticais.
# Tamanhos 3, 5, 7, 9: a abertura remove tudo que for mais estreito que k.
# Fechamento prévio de 5 px (só para o Canny): o Canny produz bordas de 1 px, e
#   uma abertura direta com k >= 3 apagaria TODAS elas, inclusive os defeitos.
#   O fechamento agrupa bordas próximas em regiões e só então a abertura atua.
# Efeito medido (Canny 15/40), FP médio por imagem boa -> fração de defeitos com
#   >= 30 px detectados: k=0: 2476 -> 0.72 | k=3: 247 -> 0.42 | k=5: 114 -> 0.22 |
#   k=7: 31 -> 0.12 | k=9: 0 -> 0.02.
#   A abertura reduz os falsos positivos, mas remove os defeitos quase na mesma
#   proporção, porque arranhões e rachaduras também são estruturas finas.
MORF_TAMANHOS = (3, 5, 7, 9)
MORF_FECHAMENTO_PREVIO = 5

# --- ROI da cápsula -------------------------------------------------------------
# Erosão de 9 px na máscara da cápsula: remove o contorno externo, que é sempre
#   detectado pelo Canny/Sobel e é sempre falso positivo. [Fração média do GT
#   dentro da ROI: erosão 0 -> 0.95 | 9 -> 0.93 | 21 -> 0.90 | 31 -> 0.87.]
#   Rachaduras e "squeeze" ficam no contorno (ex.: crack/000), então erodir mais
#   passa a esconder defeitos.
ROI_EROSAO = 9

# --- Golden template ------------------------------------------------------------
# Limiar = 30 ≈ percentil 99 da diferença |boa - referência| em 20 cápsulas boas
#   (ver calibrar_limiar_golden). [Fração de defeitos detectados / FP por imagem
#   boa: limiar 20 -> 0.83 / 6200 | 30 -> 0.63 / 2800 | 40 -> 0.42 / 1000.]
#   O p99.9 (~48) é conservador demais e perde as falhas de impressão.
# Abertura de 5 px: remove as "lascas" de 1-3 px ao longo das bordas causadas pelo
#   erro residual de alinhamento. É aqui que a abertura funciona bem, porque o
#   ruído a remover é mais fino que os defeitos.
GOLDEN_LIMIAR = 30
GOLDEN_ABERTURA = 5
GOLDEN_N_BANCO = 30        # nº de cápsulas boas candidatas a referência
ECC_ESCALA = 0.25          # alinhamento em 1/4 da resolução (rápido e mais robusto ao ruído)
ECC_ITERACOES = 200
ECC_EPS = 1e-6


# =============================================================================
# E/S
# =============================================================================
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


# =============================================================================
# 1. PRÉ-PROCESSAMENTO
# =============================================================================
def converter_cinza(img_bgr: np.ndarray) -> np.ndarray:
    """BGR -> tons de cinza (luminância ITU-R BT.601).

    Observação: a metade preta da cápsula fica com intensidade ~20-40, então defeitos
    escuros ali têm contraste muito baixo em cinza. Se isso limitar o recall, vale
    testar o canal L do Lab com CLAHE só na metade escura.
    """
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def filtro_bilateral(cinza: np.ndarray,
                     d: int = BILATERAL_D,
                     sigma_color: float = BILATERAL_SIGMA_COLOR,
                     sigma_space: float = BILATERAL_SIGMA_SPACE) -> np.ndarray:
    """Suaviza a superfície preservando bordas (peso espacial x peso de intensidade).

    Ao contrário do Gaussiano, o bilateral não "espalha" a borda de um arranhão,
    porque pixels do outro lado da borda têm intensidade muito diferente e recebem
    peso ~0. Isso mantém o gradiente do defeito alto para o Canny/Sobel.
    """
    return cv2.bilateralFilter(cinza, d, sigma_color, sigma_space)


def segmentar_capsula(cinza: np.ndarray, erosao: int = ROI_EROSAO) -> tuple[np.ndarray, np.ndarray]:
    """Máscara do objeto (cápsula) por Otsu + maior componente + preenchimento.

    Retorna (mascara_cheia, roi_interna). A ROI interna é a máscara erodida, usada
    para descartar o contorno externo da cápsula nos mapas de borda.

    Passos:
      - Otsu invertido: o fundo é claro (~200) e a cápsula é escura/laranja (< 150).
      - Fechamento 15x15: fecha os buracos do texto branco "500" e dos reflexos,
        que o Otsu classifica como fundo por serem claros.
      - Maior componente conexo + preenchimento do contorno externo: elimina
        sujeiras do fundo e garante uma máscara sólida.
    """
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
    """(Opcional/diagnóstico) Pixels de brilho especular: V alto e S baixo em HSV.

    ATENÇÃO: a tinta branca do "500"/"actavis" tem a MESMA assinatura (branco, sem
    saturação). Usar esta máscara para apagar regiões (ex.: cv2.inpaint) também
    apaga a impressão e esconde defeitos da classe faulty_imprint. Use para
    visualizar/quantificar quantos falsos positivos caem em reflexos, não para
    remover pixels cegamente.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return np.where((hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max), 255, 0).astype(np.uint8)


# =============================================================================
# 2. DETECÇÃO DE BORDAS
# =============================================================================
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
    """Magnitude do gradiente de Sobel + limiar fixo.

    Retorna (magnitude_uint8_para_visualizacao, bordas_binarias). O limiar é
    aplicado na magnitude bruta (não normalizada), para que o mesmo valor tenha o
    mesmo significado físico em todas as imagens, o que uma normalização min-max
    por imagem não garante.

    Diferente do Canny, o Sobel não faz supressão de não-máximos: as bordas saem
    "grossas" (2-6 px), então a abertura morfológica pode ser aplicada diretamente.
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)

    bordas = mag > limiar
    if roi is not None:
        bordas &= roi > 0
    mag_vis = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mag_vis, np.where(bordas, 255, 0).astype(np.uint8)


# =============================================================================
# 3. MORFOLOGIA MATEMÁTICA
# =============================================================================
def aplicar_abertura(binaria: np.ndarray,
                     tamanhos: tuple[int, ...] = MORF_TAMANHOS,
                     forma: int = cv2.MORPH_ELLIPSE,
                     fechamento_previo: int = 0) -> dict[int, np.ndarray]:
    """Abertura (erosão seguida de dilatação) para cada tamanho de elemento estruturante.

    A abertura com elemento de diâmetro k remove qualquer estrutura mais estreita
    que k em todas as direções, e preserva (aproximadamente) a forma das que são
    mais largas.

    fechamento_previo > 0 aplica um fechamento antes da abertura. Isso é
    necessário para a saída do Canny (bordas de 1 px), que seria totalmente
    apagada por uma abertura pura.

    Retorna {tamanho: mascara_resultante}.
    """
    base = binaria
    if fechamento_previo > 0:
        ef = cv2.getStructuringElement(forma, (fechamento_previo, fechamento_previo))
        base = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, ef)

    resultados = {}
    for k in tamanhos:
        ee = cv2.getStructuringElement(forma, (k, k))
        resultados[k] = cv2.morphologyEx(base, cv2.MORPH_OPEN, ee)
    return resultados


# =============================================================================
# 4. SEGMENTAÇÃO POR GOLDEN TEMPLATE
# =============================================================================
def alinhar_imagem(teste: np.ndarray, referencia: np.ndarray,
                   escala: float = ECC_ESCALA) -> tuple[np.ndarray, np.ndarray]:
    """Alinha `teste` à `referencia` (cinza) por ECC com modelo euclidiano (rotação + translação).

    O modelo euclidiano é suficiente porque câmera e distância são fixas no MVTec:
    não há variação de escala nem perspectiva, só pequenos deslocamentos e
    rotações da cápsula. O ECC é estimado em resolução reduzida (mais rápido e
    menos sensível ao ruído) e a translação é reescalada para a resolução cheia.

    Retorna (teste_alinhado, matriz_2x3).
    """
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
    """Pré-processa (cinza + bilateral) as cápsulas boas candidatas a golden template.

    Cada item guarda a imagem filtrada e a máscara cheia da cápsula (usada como ROI
    e para medir a similaridade).
    """
    banco = []
    for c in caminhos:
        img = filtro_bilateral(converter_cinza(carregar_imagem(c)))
        mascara, _ = segmentar_capsula(img, erosao=0)
        banco.append({"imagem": img, "mascara": mascara, "caminho": c})
    return banco


def selecionar_referencia(teste: np.ndarray, banco: list[dict]) -> tuple[dict, np.ndarray]:
    """Escolhe a referência do banco mais parecida com a imagem de teste.

    Uma única referência fixa (ou a mediana de várias) não funciona bem para
    capsule: a cápsula gira em torno do PRÓPRIO EIXO entre as amostras, então a
    posição vertical do "500" e a visibilidade do logo "actavis" mudam, e nenhuma
    transformação 2D corrige isso (a mediana resultante tem o "500" borrado/duplo).
    Por isso, alinhamos o teste a cada candidata e ficamos com a de menor diferença
    média dentro da cápsula, ou seja, a que tem a impressão na posição mais
    parecida.

    Retorna (item_do_banco, teste_alinhado_a_ele).
    """
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
    """|teste - referência| -> limiar -> abertura -> restrição à ROI.

    Ambas as entradas devem estar em cinza, filtradas da mesma forma e alinhadas.
    Como a impressão "500"/"actavis" também está na referência, ela se cancela na
    subtração. Esse é o principal motivo para preferir esta abordagem às bordas
    quando o problema são as letras: as bordas da impressão aparecem igualmente em
    cápsulas boas e defeituosas, enquanto aqui só sobra o que difere da referência.

    Retorna (diferenca_uint8, mascara_anomalia).
    """
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
    """Escolhe o limiar do golden template a partir de imagens SEM defeito.

    Cada cápsula boa passa exatamente pelo mesmo procedimento de uma imagem de
    teste (seleção de referência + alinhamento + diferença), e as diferenças
    dentro da cápsula são acumuladas. O percentil escolhido dessa distribuição é o
    nível de diferença que cápsulas normais produzem naturalmente (reflexos,
    desalinhamento residual, variação da impressão). Acima dele, assume-se
    anomalia. Isso dá uma justificativa estatística ao limiar, em vez de um valor
    escolhido a olho.

    Use imagens de train/good que NÃO estejam no banco de referências.
    """
    amostras = []
    for c in caminhos_bons:
        img = filtro_bilateral(converter_cinza(carregar_imagem(c)))
        ref, alinhada = selecionar_referencia(img, banco)
        amostras.append(cv2.absdiff(alinhada, ref["imagem"])[ref["mascara"] > 0])
    return int(np.ceil(np.percentile(np.concatenate(amostras), percentil)))


# =============================================================================
# MÉTRICAS PIXEL A PIXEL
# =============================================================================
def metricas_pixel(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Precisão, recall, Dice, IoU e contagem de falsos positivos (em pixels).

    Para mapas de BORDA, o recall é naturalmente baixo (a borda tem 1-3 px e a
    máscara GT é a região cheia do defeito). Para comparar etapas morfológicas,
    o indicador mais informativo é FP (pixels marcados fora do defeito).
    """
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


# =============================================================================
# VISUALIZAÇÃO
# =============================================================================
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
    """Grid do passo a passo.

    Linha 1: Original (+GT) -> Cinza -> Bilateral -> ROI
    Linha 2: Canny -> abertura k1 -> k2 -> ...
    Linha 3: Sobel -> abertura k1 -> k2 -> ...
    """
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


# =============================================================================
# ORQUESTRAÇÃO
# =============================================================================
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
    parser = argparse.ArgumentParser(description="Etapa 1 - localização de defeitos (capsule)")
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
