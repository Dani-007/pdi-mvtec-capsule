"""
Etapa 7 - Um especialista por defeito (orientação do professor).

Em vez de um método único que tenta achar todos os defeitos (golden template das Etapas 1-5),
cada defeito tem um método próprio, que procura só a marca que aquele defeito deixa na imagem,
como foi feito na Etapa 4 com a impressão "500". Cada especialista devolve UM número (quanto
maior, mais suspeita) e uma máscara com a região que ele acusou:

    1. squeeze (amassado)       a SILHUETA muda. Compara a máscara da cápsula com a silhueta boa
                                mais parecida (XOR) e mede a maior região que sobra.
    2. faulty_imprint           o "500" muda. Casa cada algarismo com o de cápsulas boas
       (impressão falha)        (correlação normalizada) e fica com o pior algarismo.
    3. crack (rachadura)        a tampa preta lasca e deixa o laranja à mostra (ou sai pó branco).
                                Segmenta preto x laranja pela COR (Otsu no canal a* do Lab) e
                                compara a forma da tampa com a de cápsulas boas (fora do "500").
    4. poke (furinho)           mancha COMPACTA e arredondada. Laplaciano da Gaussiana (LoG) em 3
                                escalas acha os "blobs"; Otsu num círculo em volta de cada pico
                                segmenta a mancha; a circularidade separa o furo de riscos e
                                pontos finos. Valor = resposta do LoG x circularidade.
    5. scratch (arranhão)       feixe de riscos finos e claros, de baixo contraste. Top-hat de
                                9 px com limiar baixo; conta os fragmentos numa janela de 101 px.

Peças comuns:
    - Contexto: tudo o que se calcula uma vez por imagem (máscara da cápsula, tampa preta,
      divisa preto/laranja, "500", logo e a zona válida para procurar contraste local).
    - Referencias: o que vem das 30 cápsulas boas do conjunto 'banco' (silhuetas, tampas,
      modelos do "500" e os envelopes).
    - Envelope: as cápsulas boas também têm estruturas claras/escuras finas (a linha de reflexo
      no meio, a marca do molde, linhas internas), sempre no mesmo lugar. O envelope é o MÁXIMO
      do top-hat (e do LoG, no especialista 4) nas 30 boas do banco, dilatado 9 px (tolerância
      de posição). Só o que passa do envelope conta como suspeito. O "500" e o logo mudam de
      lugar (a cápsula gira), então ficam fora da zona válida: o "500" é do especialista 2 e
      o logo é variação normal.

A pose da cápsula é estável na base (centro +-3 px, ângulo +-0,3 grau, tampa preta sempre à
esquerda), então o alinhamento com as referências é só uma translação pelo centroide.

Nada aqui usa rótulos. Os limiares de decisão ficam em cascata.py (definidos só no treino).

Uso (passo a passo de uma imagem):
    python src/especialistas.py --defeito crack --indice 10 --salvar saida.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from impressao import segmentar_impressao, segmentar_logo
from pipeline import (
    RAIZ, carregar_imagem, carregar_mascara_gt, caminho_teste, converter_cinza,
    filtro_bilateral, segmentar_capsula,
)

# =============================================================================
# PARÂMETROS
# =============================================================================
ESTAGIOS = ["squeeze", "faulty_imprint", "crack", "poke", "scratch"]   # ordem da cascata
NOMES = {"squeeze": "amassado", "faulty_imprint": "impressão falha", "crack": "rachadura",
         "poke": "furinho", "scratch": "arranhão"}

# --- Zona válida (onde os especialistas 4 e 5 procuram contraste local) --------
# Faixa de 13 px junto ao contorno: a borda da cápsula tem contraste alto e pequenas variações
#   de forma viram risco claro/escuro no top-hat. (Com 19 ou 25 px a detecção piora: rachaduras
#   e furinhos ficam perto da borda.)
# "500" dilatado 13 px: a posição da impressão muda com o giro; quem cuida dela é o
#   especialista 2. (Excluir o retângulo inteiro do "500" piora: arranhões e furinhos passam
#   entre os algarismos.)
# Divisa preto/laranja: faixa de +-9 px em torno da coluna da divisa (degrau forte e posição
#   variável, porque o comprimento da cápsula varia ~30 px).
BANDA_CONTORNO = 13
DILATA_IMPRESSAO = 13
MEIA_LARGURA_DIVISA = 9
LIMIAR_PRETO = 60          # cinza < 60 = tampa preta (mesmo valor de impressao.py)

# --- Logo "actavis" --------------------------------------------------------------
# O logo aparece em ~1/3 das cápsulas, em altura variável (a tampa gira). Ele é uma CADEIA
# horizontal comprida de componentes claros; um furinho ou uma lasca são manchas isoladas.
# Fechamento com retângulo 45x15 junta as letras; só cadeias com >= 100 px de largura são
# tratadas como logo (um furinho de 80 px continua visível).
LOGO_FECHAMENTO = (45, 15)
LOGO_LARGURA_MIN = 100
LOGO_DILATA = 15

# --- Envelope das cápsulas boas ---------------------------------------------------
TOLERANCIA_ENVELOPE = 9    # dilatação (px) de cada mapa bom: tolera +-4 px de posição

# --- 1. Squeeze --------------------------------------------------------------------
# Abertura 7x7 no XOR das silhuetas: entre duas cápsulas boas sobra uma casca de 1-3 px ao
#   longo do contorno; a abertura apaga a casca e mantém as regiões largas (amassados).
SQUEEZE_ABERTURA = 7

# --- 2. Impressão ------------------------------------------------------------------
IMP_MARGEM = 8             # borda (px) em volta do "500" no modelo
IMP_BUSCA = (40, 60)       # busca do "500" inteiro: +-40 px em x, +-60 px em y (giro)
IMP_BUSCA_ALGARISMO = 6    # ajuste fino de cada algarismo (o giro "achata" a impressão)
IMP_TOLERANCIA_LARGURA = 6 # modelos válidos: largura do "500" a até 6 px da mediana do banco
IMP_LIMIAR_MASCARA = 30    # |teste - modelo| > 30 marca a região (mesmo limiar da Etapa 1)

KMEANS_SEMENTE = 42        # candidato k-means: sementes fixas, para repetir o resultado

# --- 3. Rachadura ------------------------------------------------------------------
# Abertura 9x9 no XOR das tampas: a borda da região laranja é menos nítida que a silhueta
#   (sombra na base da cápsula), então a casca normal é mais grossa que no squeeze.
RACHADURA_ABERTURA = 9

# --- 4. Furinho: LoG + Otsu no círculo + circularidade (sugestão do professor) -------
# O LoG (sigma² x Laplaciano da imagem suavizada por uma Gaussiana) é um detector de "blobs":
#   responde no centro de manchas arredondadas de raio ~ sigma x raiz(2) e pouco a riscos finos
#   (a resposta a um risco de largura w cai com w/sigma). Por isso ele já separa o furo do
#   arranhão na detecção.
# Escalas 6, 9 e 13 px: blobs de ~17, 25 e 37 px de diâmetro (os furos têm 30-60 px, com
#   detalhes internos de 5-15 px). Medido com cada escala sozinha, furinhos acima do percentil
#   98 das boas: sigma 4 -> 6 de 21; 6 -> 10; 9 -> 11; 13 -> 12; 18 -> 8. Com sigma = 4 entram
#   os pontos de poeira; com 13 sozinho, 15 dos 20 amassados também são acusados.
# Raio do círculo = 3 sigma: o blob de raio 1,4 sigma ocupa ~1/4 da janela, e o Otsu separa
#   bem duas classes quando a menor não é minúscula.
# Suporte mínimo 0,5: o LoG só vale onde pelo menos metade do peso da Gaussiana cai em pixels
#   da zona de busca (evita tiras estreitas entre o "500" e o contorno).
# Margem de 6 px em volta do recorte do "500" (que já tem 8 px de borda): dentro do retângulo
#   da impressão quem responde é o especialista 2. A máscara do "500" sozinha não basta: em
#   parte das boas ela perde um pedaço do "5", que vira um blob claro de ~40 px.
# Área mínima 150 px (disco de ~14 px): os pontos de poeira das cápsulas boas são blobs
#   perfeitos (circularidade ~0,9), mas pequenos: ~80-110 px. Com 100 px a etapa volta a
#   acusar 3 arranhões (em vez de 1); com 200 ou 250 px a cascata dá o mesmo resultado.
# Fechamento 5 + abertura 3 na máscara do Otsu: tapa os pontos escuros dentro do furo e tira
#   os pixels soltos antes de medir o perímetro.
FURINHO_SIGMAS = (6, 9, 13)
FURINHO_RAIO_EM_SIGMAS = 3
FURINHO_SUPORTE_MIN = 0.5
FURINHO_MARGEM_500 = 6
FURINHO_CANDIDATOS = 5     # picos do LoG examinados por imagem
FURINHO_AREA_MIN = 150
FURINHO_FECHAMENTO = 5
FURINHO_ABERTURA = 3

# Candidato "crescimento de região" (segmentação por regiões, no lugar do Otsu no círculo):
#   a partir do pico do LoG, entram os vizinhos com cinza a até 10 níveis do pico (o ruído
#   tem 5-15 níveis e as bordas de defeito >= 30). Testado com 6, 10 e 15: 5, 8 e 9 furinhos
#   acima do percentil 98 das boas, contra 12 do Otsu no círculo.
FURINHO_TOLERANCIA_REGIAO = 10

# Método anterior do furinho (mantido como candidato na tabela de comparação): top-hat e
#   black-hat de 15 px acima do envelope, limiar 6, fechamento 5 + abertura 3, maior mancha.
TOPHAT_FURINHO_K = 15
TOPHAT_FURINHO_LIMIAR = 6

# --- 5. Arranhão -------------------------------------------------------------------
# Os riscos têm 2-5 px de largura e só 3-10 níveis de cinza de contraste. Por isso: elemento
#   pequeno (9 px), limiar baixo (3) e imagem SEM o filtro bilateral (ele apaga os riscos;
#   aqui só um Gaussiano 5x5). Com limiar tão baixo o risco sai picotado em fragmentos; o que
#   denuncia o arranhão é a CONCENTRAÇÃO de fragmentos, medida numa janela de 101 px.
ARRANHAO_K = 9
ARRANHAO_LIMIAR = 3
ARRANHAO_AREA_MIN = 4      # fragmentos menores que 4 px são ruído
ARRANHAO_JANELA = 101


def _ee(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _deslocar(img: np.ndarray, dx: float, dy: float, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Translação (dx, dy); o que entra pela borda é 0."""
    h, w = img.shape[:2]
    return cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=interp)


def _centroide(mascara: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(mascara, binaryImage=True)
    if m["m00"] == 0:
        return mascara.shape[1] / 2, mascara.shape[0] / 2
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def _maior_componente(mascara: np.ndarray) -> float:
    """Área (px) do maior componente conexo da máscara (0 se vazia)."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    return float(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0.0


def _caixa(mascara: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mascara > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# =============================================================================
# CONTEXTO: o que se calcula uma vez por imagem
# =============================================================================
@dataclass
class Contexto:
    bgr: np.ndarray
    cinza: np.ndarray          # tons de cinza + bilateral (Etapa 1)
    suave: np.ndarray          # tons de cinza + Gaussiano 5x5 (preserva riscos finos)
    capsula: np.ndarray        # máscara da cápsula (silhueta)
    centro: tuple[float, float]
    preto: np.ndarray          # tampa preta: cápsula menos a região laranja
    centro_preto: tuple[float, float]
    x_divisa: int              # coluna da divisa preto/laranja
    impressao: np.ndarray      # máscara do "500"
    logo: np.ndarray           # zona do logo (vazia se ele não aparece)
    valida: np.ndarray         # bool: onde os especialistas 4 e 5 procuram
    cache: dict = field(default_factory=dict)   # resultados reusados entre especialistas


def segmentar_laranja(bgr: np.ndarray, capsula: np.ndarray, canal: str = "a") -> np.ndarray:
    """Região laranja da cápsula pelo histograma de COR dentro da cápsula (Otsu).

    canal 'a' (padrão): a* do Lab, que mede verde-vermelho. O laranja tem a* alto; o preto, o
    logo cinza e o pó branco têm a* baixo, então os três caem no lado "não laranja" e o logo
    não fura a tampa preta. canal 'cinza' é o candidato descartado: o limiar de Otsu no cinza
    cai entre o logo e o laranja escuro da borda, e a forma da tampa passa a depender do logo.
    canal 'kmeans' é outro candidato descartado: k-means com 2 grupos no plano (a*, b*); o
    grupo de a* maior é o laranja. Dá quase a mesma máscara do Otsu no a*, sem ganho.
    """
    if canal == "kmeans":
        dentro = capsula > 0
        if not dentro.any():
            return np.zeros_like(capsula)
        lab = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB), (5, 5), 0)
        cv2.setRNGSeed(KMEANS_SEMENTE)
        _, rotulos, centros = cv2.kmeans(
            lab[dentro][:, 1:3].astype(np.float32), 2, None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3, cv2.KMEANS_PP_CENTERS)
        laranja = np.zeros_like(capsula)
        laranja[dentro] = np.where(rotulos.ravel() == int(np.argmax(centros[:, 0])), 255, 0)
        return laranja
    if canal == "a":
        c = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)[..., 1]
    else:
        c = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    c = cv2.GaussianBlur(c, (5, 5), 0)
    dentro = capsula > 0
    if not dentro.any():
        return np.zeros_like(capsula)
    t, _ = cv2.threshold(c[dentro].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.where((c > t) & dentro, 255, 0).astype(np.uint8)


def coluna_divisa(cinza: np.ndarray, capsula: np.ndarray) -> int:
    """Coluna seguinte à última em que mais da metade da cápsula é preta.
    Na base, a tampa preta está sempre à esquerda.

    A regra anterior (primeira coluna, da esquerda para a direita, com menos da metade preta)
    parava na ponta arredondada da tampa em 17 das 351 imagens: o brilho da ponta deixa as
    primeiras colunas com cinza >= 60, a divisa saía em x ~ 85 em vez de ~ 515 e a faixa da
    divisa não era tirada da zona válida. Nas outras 334 imagens as duas regras dão a mesma
    coluna."""
    dentro = capsula > 0
    altura = dentro.sum(0)
    frac_preta = ((cinza < LIMIAR_PRETO) & dentro).sum(0) / np.maximum(altura, 1)
    xs = np.where(altura > 100)[0]
    if len(xs) == 0:
        return cinza.shape[1] // 2
    pretas = xs[frac_preta[xs] >= 0.5]
    return int(pretas[-1]) + 1 if len(pretas) else int(xs[len(xs) // 2])


def zona_logo(cinza: np.ndarray, capsula: np.ndarray) -> np.ndarray:
    """Zona do logo: cadeia horizontal comprida de componentes claros na tampa preta."""
    logo = segmentar_logo(cinza, capsula)
    if not logo.any():
        return logo
    junto = cv2.morphologyEx(logo, cv2.MORPH_CLOSE,
                             cv2.getStructuringElement(cv2.MORPH_RECT, LOGO_FECHAMENTO))
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(junto, connectivity=8)
    saida = np.zeros_like(logo)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_WIDTH] >= LOGO_LARGURA_MIN:
            saida[rotulos == i] = 255
    return cv2.dilate(saida, _ee(LOGO_DILATA))


def preparar(bgr: np.ndarray) -> Contexto:
    bruto = converter_cinza(bgr)
    cinza = filtro_bilateral(bruto)
    capsula, _ = segmentar_capsula(cinza, erosao=0)
    laranja = segmentar_laranja(bgr, capsula)
    preto = np.where((capsula > 0) & (laranja == 0), 255, 0).astype(np.uint8)
    x_div = coluna_divisa(cinza, capsula)
    impressao, _ = segmentar_impressao(cinza, capsula)
    logo = zona_logo(cinza, capsula)

    valida = cv2.erode(capsula, _ee(2 * BANDA_CONTORNO + 1)) > 0
    valida &= cv2.dilate(impressao, _ee(DILATA_IMPRESSAO)) == 0
    valida &= logo == 0
    valida[:, max(x_div - MEIA_LARGURA_DIVISA, 0):x_div + MEIA_LARGURA_DIVISA + 1] = False

    return Contexto(bgr=bgr, cinza=cinza, suave=cv2.GaussianBlur(bruto, (5, 5), 0),
                    capsula=capsula, centro=_centroide(capsula), preto=preto,
                    centro_preto=_centroide(preto), x_divisa=x_div, impressao=impressao,
                    logo=logo, valida=valida)


def realce_claro(ctx: Contexto, k: int) -> np.ndarray:
    """Top-hat: estruturas CLARAS mais estreitas que k px."""
    return cv2.morphologyEx(ctx.suave, cv2.MORPH_TOPHAT, _ee(k))


def realce_escuro(ctx: Contexto, k: int) -> np.ndarray:
    """Black-hat: estruturas ESCURAS mais estreitas que k px."""
    return cv2.morphologyEx(ctx.suave, cv2.MORPH_BLACKHAT, _ee(k))


# =============================================================================
# REFERÊNCIAS: o que vem das cápsulas boas do banco
# =============================================================================
@dataclass
class Referencias:
    silhuetas: list[tuple[np.ndarray, tuple[float, float]]]
    tampas: list[tuple[np.ndarray, tuple[float, float]]]
    modelos_500: list[dict]
    centro_medio: tuple[float, float]
    envelopes: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)
    area_500: float = 1.0      # área mediana do "500" no banco (só para candidatos)
    forma: tuple[np.ndarray, np.ndarray] | None = None   # média/desvio dos descritores globais


def _modelos_500(ctxs: list[Contexto]) -> list[dict]:
    """Recorte do "500" de cada cápsula boa (com margem) e os 3 trechos dos algarismos.

    Em parte das boas a máscara do "500" gruda na linha de reflexo e fica larga demais;
    essas não servem de modelo (a divisão em 3 algarismos ficaria errada)."""
    caixas = [_caixa(c.impressao) for c in ctxs]
    larguras = [cx[2] - cx[0] for cx in caixas if cx is not None]
    mediana = float(np.median(larguras))
    modelos = []
    for c, cx in zip(ctxs, caixas):
        if cx is None or abs((cx[2] - cx[0]) - mediana) > IMP_TOLERANCIA_LARGURA:
            continue
        x0, y0, x1, y1 = cx
        m = IMP_MARGEM
        terco = (x1 - x0) / 3
        cortes = [(int(k * terco) + m - 4, int((k + 1) * terco) + m + 4) for k in range(3)]
        modelos.append({"imagem": c.cinza[y0 - m:y1 + m, x0 - m:x1 + m].copy(),
                        "x": x0 - m, "y": y0 - m, "cortes": cortes})
    return modelos


def construir_referencias(caminhos: list[Path],
                          ks_claro: tuple[int, ...] = (ARRANHAO_K, TOPHAT_FURINHO_K),
                          ks_escuro: tuple[int, ...] = (TOPHAT_FURINHO_K,),
                          tolerancia: int = TOLERANCIA_ENVELOPE) -> Referencias:
    """Monta as referências com as cápsulas boas do banco (30 imagens, ~20 s).

    ks_claro/ks_escuro: tamanhos dos envelopes de top-hat/black-hat (arranhão e método
    anterior do furinho, que cascata.py mantém para a tabela "antes x depois"). Os envelopes
    do LoG (especialista 4) são sempre montados."""
    ctxs = [preparar(carregar_imagem(c)) for c in caminhos]
    centro = tuple(np.mean([c.centro for c in ctxs], axis=0))
    refs = Referencias(
        silhuetas=[(c.capsula, c.centro) for c in ctxs],
        tampas=[(c.preto, c.centro_preto) for c in ctxs],
        modelos_500=_modelos_500(ctxs),
        centro_medio=centro,
        area_500=float(np.median([np.count_nonzero(c.impressao) for c in ctxs])),
    )
    descr = np.array([descritores_silhueta(c.capsula) for c in ctxs])
    refs.forma = (descr.mean(0), descr.std(0))

    def envelope(funcao, k: int) -> np.ndarray:
        mapas = []
        for c in ctxs:
            m = funcao(c, k) * c.valida.astype(np.uint8)
            m = cv2.dilate(m, _ee(tolerancia))
            mapas.append(_deslocar(m, centro[0] - c.centro[0], centro[1] - c.centro[1]))
        return np.max(mapas, axis=0)

    for k in ks_claro:
        refs.envelopes[("claro", k)] = envelope(realce_claro, k)
    for k in ks_escuro:
        refs.envelopes[("escuro", k)] = envelope(realce_escuro, k)

    # Envelopes do LoG: o máximo, nas boas, da resposta a blobs claros e escuros em cada escala
    for c in ctxs:
        for chave, m in mapas_log(c, zona_blob(c, refs)).items():
            m = _deslocar(cv2.dilate(m, _ee(tolerancia)),
                          centro[0] - c.centro[0], centro[1] - c.centro[1])
            refs.envelopes[chave] = m if chave not in refs.envelopes else np.maximum(
                refs.envelopes[chave], m)
        c.cache.clear()
    return refs


def acima_do_envelope(ctx: Contexto, refs: Referencias, tipo: str, k: int,
                      usar_envelope: bool = True) -> np.ndarray:
    """Realce (top-hat ou black-hat) da imagem menos o envelope das boas, na zona válida."""
    mapa = realce_claro(ctx, k) if tipo == "claro" else realce_escuro(ctx, k)
    if usar_envelope:
        env = _deslocar(refs.envelopes[(tipo, k)], ctx.centro[0] - refs.centro_medio[0],
                        ctx.centro[1] - refs.centro_medio[1])
        mapa = cv2.subtract(mapa, env)
    return mapa * ctx.valida.astype(np.uint8)


# =============================================================================
# 1. SQUEEZE: silhueta
# =============================================================================
def _xor_mais_parecida(mascara: np.ndarray, centro: tuple[float, float],
                       referencias: list[tuple[np.ndarray, tuple[float, float]]]) -> np.ndarray:
    """XOR da máscara com a referência mais parecida (a de menor XOR), alinhada pelo centroide.
    Usar a mais parecida, e não uma silhueta média, tolera a variação normal de comprimento."""
    melhor, menor = None, None
    for ref, c in referencias:
        x = cv2.bitwise_xor(mascara, _deslocar(ref, centro[0] - c[0], centro[1] - c[1],
                                               cv2.INTER_NEAREST))
        n = cv2.countNonZero(x)
        if menor is None or n < menor:
            melhor, menor = x, n
    return melhor


def especialista_squeeze(ctx: Contexto, refs: Referencias,
                         abertura: int = SQUEEZE_ABERTURA) -> tuple[float, np.ndarray]:
    xor = _xor_mais_parecida(ctx.capsula, ctx.centro, refs.silhuetas)
    mascara = cv2.morphologyEx(xor, cv2.MORPH_OPEN, _ee(abertura))
    return _maior_componente(mascara), mascara


def descritores_silhueta(capsula: np.ndarray) -> list[float]:
    """Descritores globais da silhueta: área, solidez, altura e compacidade."""
    contornos, _ = cv2.findContours(capsula, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return [0.0, 0.0, 0.0, 0.0]
    c = max(contornos, key=cv2.contourArea)
    area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
    casco = cv2.contourArea(cv2.convexHull(c))
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return [area, area / max(casco, 1), min(w, h), perim ** 2 / (4 * np.pi * max(area, 1))]


def candidato_squeeze_descritores(ctx: Contexto, refs: Referencias) -> float:
    """Candidato descartado: maior z-score dos descritores globais em relação ao banco."""
    media, desvio = refs.forma
    z = np.abs(np.array(descritores_silhueta(ctx.capsula)) - media) / np.maximum(desvio, 1e-9)
    return float(z.max())


# =============================================================================
# 2. IMPRESSÃO: casamento de cada algarismo
# =============================================================================
def _casar_500(ctx: Contexto, modelo: dict) -> tuple[float, float, tuple[int, int]]:
    """Casa o "500" do modelo na imagem. Devolve (correlação do "500" inteiro, correlação do
    pior algarismo, posição (x, y) do casamento)."""
    img = ctx.cinza
    t = modelo["imagem"]
    th, tw = t.shape
    bx, by = IMP_BUSCA
    x0, y0 = max(modelo["x"] - bx, 0), max(modelo["y"] - by, 0)
    janela = img[y0:modelo["y"] + th + by, x0:modelo["x"] + tw + bx]
    if janela.shape[0] < th or janela.shape[1] < tw:
        return -1.0, -1.0, (modelo["x"], modelo["y"])
    _, inteiro, _, pos = cv2.minMaxLoc(cv2.matchTemplate(janela, t, cv2.TM_CCOEFF_NORMED))
    px, py = x0 + pos[0], y0 + pos[1]

    pior = 1.0
    f = IMP_BUSCA_ALGARISMO
    for a, b in modelo["cortes"]:
        a, b = max(a, 0), min(b, tw)
        alg = t[:, a:b]
        jy, jx = max(py - f, 0), max(px + a - f, 0)
        jan = img[jy:py + th + f, jx:px + b + f]
        if jan.shape[0] < alg.shape[0] or jan.shape[1] < alg.shape[1]:
            return inteiro, -1.0, (px, py)
        pior = min(pior, cv2.minMaxLoc(cv2.matchTemplate(jan, alg, cv2.TM_CCOEFF_NORMED))[1])
    return inteiro, pior, (px, py)


def _casamentos_500(ctx: Contexto, refs: Referencias) -> list[tuple]:
    """(modelo, correlação do "500" inteiro, correlação do pior algarismo, posição) para cada
    modelo do banco. Calculado uma vez por imagem: os especialistas 2 e 4 usam."""
    if "casamentos_500" not in ctx.cache:
        ctx.cache["casamentos_500"] = [(m, *_casar_500(ctx, m)) for m in refs.modelos_500]
    return ctx.cache["casamentos_500"]


def especialista_impressao(ctx: Contexto, refs: Referencias, por_algarismo: bool = True,
                           abrir: bool = True) -> tuple[float, np.ndarray]:
    """1 - correlação do pior algarismo, com o modelo do banco que melhor casa.

    por_algarismo=False é o candidato descartado (correlação do "500" inteiro): uma falha num
    algarismo só muda 1/3 da área casada e a correlação cai pouco. abrir=False deixa a
    máscara sem a abertura 3x3 (só para medir o efeito da morfologia)."""
    melhor, modelo_m, pos_m = -1.0, None, (0, 0)
    for modelo, inteiro, pior, pos in _casamentos_500(ctx, refs):
        valor = pior if por_algarismo else inteiro
        if valor > melhor:
            melhor, modelo_m, pos_m = valor, modelo, pos
    mascara = np.zeros_like(ctx.cinza)
    if modelo_m is not None:
        t = modelo_m["imagem"]
        px, py = pos_m
        recorte = ctx.cinza[py:py + t.shape[0], px:px + t.shape[1]]
        if recorte.shape == t.shape:
            dif = np.where(cv2.absdiff(recorte, t) > IMP_LIMIAR_MASCARA, 255, 0).astype(np.uint8)
            mascara[py:py + t.shape[0], px:px + t.shape[1]] = cv2.morphologyEx(
                dif, cv2.MORPH_OPEN, _ee(3)) if abrir else dif
    return 1.0 - melhor, mascara


CHAVE_FRACAO_CLARA = "dif500_frac_clara"   # coluna/limiar usados por cascata.py para o tipo


def diferenca_do_500(ctx: Contexto, refs: Referencias) -> dict[str, float]:
    """Atributos da diferença entre o "500" da cápsula e o modelo do banco que melhor casa
    (o mesmo modelo do especialista 2), para a ficha de classificacao_cascata.py.

    O sinal da diferença diz O QUE mudou na impressão: na impressão falha FALTA tinta, então
    a cápsula fica mais escura que o modelo, em cima dos traços; um furo ou uma lasca sobre o
    "500" expõe material mais claro, também fora dos traços. Tinta do modelo = Otsu no recorte
    do modelo; conta só |diferença| > IMP_LIMIAR_MASCARA, depois de uma abertura 3x3.

        dif500_area          pixels com diferença
        dif500_frac_clara    fração deles em que a cápsula está mais clara que o modelo
        dif500_claro_fora    pixels mais claros FORA dos traços do modelo
        dif500_escuro_tinta  pixels mais escuros SOBRE os traços do modelo"""
    f = {"dif500_area": 0.0, "dif500_frac_clara": 0.0, "dif500_claro_fora": 0.0,
         "dif500_escuro_tinta": 0.0}
    casos = _casamentos_500(ctx, refs)
    if not casos:
        return f
    modelo, _, _, (px, py) = max(casos, key=lambda c: c[2])
    t = modelo["imagem"]
    recorte = ctx.cinza[py:py + t.shape[0], px:px + t.shape[1]]
    if recorte.shape != t.shape:
        return f
    dif = recorte.astype(np.int16) - t.astype(np.int16)
    tinta = cv2.threshold(t, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] > 0
    mudou = np.where(np.abs(dif) > IMP_LIMIAR_MASCARA, 255, 0).astype(np.uint8)
    mudou = cv2.morphologyEx(mudou, cv2.MORPH_OPEN, _ee(3)) > 0
    if mudou.any():
        f.update({"dif500_area": float(mudou.sum()),
                  "dif500_frac_clara": float((dif[mudou] > 0).mean()),
                  "dif500_claro_fora": float((mudou & (dif > 0) & ~tinta).sum()),
                  "dif500_escuro_tinta": float((mudou & (dif < 0) & tinta).sum())})
    return f


def candidato_impressao_area(ctx: Contexto, refs: Referencias) -> float:
    """Candidato descartado (atributo da Etapa 4): desvio da área do "500" em relação à área
    típica do banco."""
    return abs(np.count_nonzero(ctx.impressao) / refs.area_500 - 1.0)


# =============================================================================
# 3. RACHADURA: forma da tampa preta (segmentação por cor)
# =============================================================================
def especialista_rachadura(ctx: Contexto, refs: Referencias, abertura: int = RACHADURA_ABERTURA,
                           tampa: np.ndarray | None = None,
                           tampas_ref: list | None = None) -> tuple[float, np.ndarray]:
    """XOR da tampa preta com a tampa boa mais parecida, fora do "500", + abertura.

    O "500" é excluído porque, quando a tinta sai mais clara que o normal, ela cai no lado
    "não laranja" do Otsu e fura a região laranja; a impressão é assunto do especialista 2.
    `tampa`/`tampas_ref` permitem testar outra segmentação (candidato com Otsu no cinza)."""
    if tampa is None:
        tampa, centro = ctx.preto, ctx.centro_preto
    else:
        centro = _centroide(tampa)
    xor = _xor_mais_parecida(tampa, centro, refs.tampas if tampas_ref is None else tampas_ref)
    xor[cv2.dilate(ctx.impressao, _ee(DILATA_IMPRESSAO)) > 0] = 0
    mascara = cv2.morphologyEx(xor, cv2.MORPH_OPEN, _ee(abertura))
    return _maior_componente(mascara), mascara


# =============================================================================
# 4. FURINHO: blobs do LoG, Otsu no círculo e circularidade
# =============================================================================
def zona_blob(ctx: Contexto, refs: Referencias) -> np.ndarray:
    """Onde o especialista 4 procura: a zona válida menos o retângulo em que o "500" casou
    com o modelo mais parecido do banco (o mesmo casamento do especialista 2)."""
    if "zona_blob" not in ctx.cache:
        zona = ctx.valida.copy()
        casos = _casamentos_500(ctx, refs)
        if casos:
            modelo, _, _, (px, py) = max(casos, key=lambda c: c[1])
            th, tw = modelo["imagem"].shape
            m = FURINHO_MARGEM_500
            zona[max(py - m, 0):py + th + m, max(px - m, 0):px + tw + m] = False
        ctx.cache["zona_blob"] = zona
    return ctx.cache["zona_blob"]


def realce_log(ctx: Contexto, zona: np.ndarray, sigma: float) -> np.ndarray:
    """Laplaciano da Gaussiana normalizado em escala: -sigma² x Laplaciano(G_sigma * imagem).
    Positivo no centro de um blob CLARO, negativo no de um blob ESCURO.

    A Gaussiana é calculada só com os pixels da zona de busca (média ponderada: soma dos
    vizinhos válidos / soma dos pesos válidos), separadamente de cada lado da divisa. Sem
    isso, os degraus fortes que não são defeito (fundo, tampa preta, "500", logo) vazam para
    dentro da zona, porque a Gaussiana de sigma = 13 alcança ~40 px."""
    img = ctx.suave.astype(np.float32)
    saida = np.zeros_like(img)
    esquerda = np.arange(img.shape[1]) < ctx.x_divisa
    for lado in (esquerda, ~esquerda):
        m = (zona & lado[None, :]).astype(np.float32)
        peso = cv2.GaussianBlur(m, (0, 0), sigma)
        media = cv2.GaussianBlur(img * m, (0, 0), sigma) / np.maximum(peso, 1e-3)
        log = -(sigma ** 2) * cv2.Laplacian(media, cv2.CV_32F, ksize=1)
        ok = (m > 0) & (peso > FURINHO_SUPORTE_MIN)
        saida[ok] = log[ok]
    return saida


def mapas_log(ctx: Contexto, zona: np.ndarray,
              sigmas: tuple[int, ...] = FURINHO_SIGMAS) -> dict[tuple[str, int], np.ndarray]:
    """Resposta a blobs claros e a blobs escuros em cada escala (mapas >= 0)."""
    mapas = {}
    for s in sigmas:
        log = realce_log(ctx, zona, s)
        mapas[("log_claro", s)] = np.maximum(log, 0)
        mapas[("log_escuro", s)] = np.maximum(-log, 0)
    return mapas


def circularidade(regiao: np.ndarray) -> tuple[float, float]:
    """(área, 4 pi A / P²) do contorno externo da região: 1 = círculo; um risco de 4 px de
    largura que atravessa uma janela de 60 px dá ~0,2."""
    contornos, _ = cv2.findContours(regiao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contornos:
        return 0.0, 0.0
    c = max(contornos, key=cv2.contourArea)
    area, perim = cv2.contourArea(c), cv2.arcLength(c, True)
    return float(area), float(min(4 * np.pi * area / max(perim ** 2, 1.0), 1.0))


def otsu_no_circulo(ctx: Contexto, zona: np.ndarray, x: int, y: int, raio: int, claro: bool,
                    morfologia: bool = True,
                    segmentacao: str = "otsu") -> tuple[np.ndarray, float, float]:
    """Segmenta a mancha em volta do pico (x, y) do LoG: Otsu dos tons de cinza dentro do
    círculo (só pixels da zona de busca, do mesmo lado da divisa). O LoG diz ONDE está o blob
    e se ele é claro ou escuro; o Otsu, limitado ao círculo, acha o limiar entre a mancha e a
    vizinhança dela (um Otsu na cápsula inteira separaria preto x laranja).

    Devolve (região, área, circularidade). A região é o componente da classe do blob que toca
    o disco central (raio/4).

    morfologia=False tira o fechamento e a abertura (só para medir o efeito da morfologia).
    segmentacao="regiao" é o candidato descartado: crescimento de região a partir do pico
    (vizinhos com cinza a até FURINHO_TOLERANCIA_REGIAO níveis do pico), dentro do círculo."""
    h, w = ctx.cinza.shape
    regiao = np.zeros((h, w), np.uint8)
    y0, y1, x0, x1 = max(y - raio, 0), min(y + raio + 1, h), max(x - raio, 0), min(x + raio + 1, w)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    janela = (dist2 <= raio ** 2) & zona[y0:y1, x0:x1] & ((xx < ctx.x_divisa) == (x < ctx.x_divisa))
    recorte = ctx.cinza[y0:y1, x0:x1]
    tons = recorte[janela]
    if len(tons) < 20 or tons.min() == tons.max():
        return regiao, 0.0, 0.0
    if segmentacao == "regiao":
        if not janela[y - y0, x - x0]:
            return regiao, 0.0, 0.0
        barreira = np.ones((recorte.shape[0] + 2, recorte.shape[1] + 2), np.uint8)
        barreira[1:-1, 1:-1] = np.where(janela, 0, 1)
        tol = FURINHO_TOLERANCIA_REGIAO
        cv2.floodFill(np.ascontiguousarray(recorte), barreira, (x - x0, y - y0), 0, tol, tol,
                      4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8))
        binaria = np.where(barreira[1:-1, 1:-1] == 255, 255, 0).astype(np.uint8)
    else:
        t, _ = cv2.threshold(tons.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        classe = (recorte > t) if claro else (recorte <= t)
        binaria = np.where(classe & janela, 255, 0).astype(np.uint8)
    if morfologia:
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, _ee(FURINHO_FECHAMENTO))
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, _ee(FURINHO_ABERTURA))
    binaria[~janela] = 0
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    centrais = np.unique(rotulos[(dist2 <= max(raio // 4, 2) ** 2) & (rotulos > 0)])
    if len(centrais) == 0:
        return regiao, 0.0, 0.0
    i = max(centrais, key=lambda j: stats[j, cv2.CC_STAT_AREA])
    regiao[y0:y1, x0:x1][rotulos == i] = 255
    area, circ = circularidade(regiao[y0:y1, x0:x1])
    return regiao, area, circ


def blobs_furinho(ctx: Contexto, refs: Referencias, sigmas: tuple[int, ...] = FURINHO_SIGMAS,
                  morfologia: bool = True,
                  segmentacao: str = "otsu") -> tuple[np.ndarray, list[dict]]:
    """Mapa do LoG acima do envelope (máximo entre escalas e entre claro/escuro) e os blobs
    candidatos: os FURINHO_CANDIDATOS maiores picos, cada um com a região do Otsu no círculo.

    Cada blob: x, y, sigma, claro, raio, resposta (pico do LoG acima do envelope), regiao,
    area, circularidade e valor = resposta x circularidade (0 se a área for menor que a
    mínima)."""
    zona = zona_blob(ctx, refs)
    dx, dy = ctx.centro[0] - refs.centro_medio[0], ctx.centro[1] - refs.centro_medio[1]
    chaves, pilha = [], []
    for chave, m in mapas_log(ctx, zona, sigmas).items():
        chaves.append(chave)
        pilha.append(np.maximum(m - _deslocar(refs.envelopes[chave], dx, dy), 0) * zona)
    pilha = np.stack(pilha)
    mapa, qual = pilha.max(0), pilha.argmax(0)

    resto = mapa.copy()
    blobs = []
    for _ in range(FURINHO_CANDIDATOS):
        _, resposta, _, (x, y) = cv2.minMaxLoc(resto)
        if resposta <= 0:
            break
        tipo, sigma = chaves[qual[y, x]]
        raio = int(round(FURINHO_RAIO_EM_SIGMAS * sigma))
        regiao, area, circ = otsu_no_circulo(ctx, zona, x, y, raio, tipo == "log_claro",
                                             morfologia, segmentacao)
        blobs.append({"x": x, "y": y, "sigma": sigma, "claro": tipo == "log_claro", "raio": raio,
                      "resposta": float(resposta), "regiao": regiao, "area": area,
                      "circularidade": circ,
                      "valor": float(resposta) * circ if area >= FURINHO_AREA_MIN else 0.0})
        cv2.circle(resto, (x, y), raio, 0, -1)   # o próximo pico vem de outro lugar
    return mapa, blobs


def especialista_furinho(ctx: Contexto, refs: Referencias, usar_forma: bool = True,
                         morfologia: bool = True,
                         segmentacao: str = "otsu") -> tuple[float, np.ndarray]:
    """Maior (resposta do LoG x circularidade) entre os blobs com área >= FURINHO_AREA_MIN; a
    máscara é a região do Otsu desse blob.

    usar_forma=False é o candidato "só o LoG": o maior pico acima do envelope, sem o Otsu no
    círculo, a circularidade e a área mínima."""
    mapa, blobs = blobs_furinho(ctx, refs, morfologia=morfologia, segmentacao=segmentacao)
    if not usar_forma:
        mascara = blobs[0]["regiao"] if blobs else np.zeros_like(ctx.cinza)
        return float(mapa.max()), mascara
    melhor = max(blobs, key=lambda b: b["valor"], default=None)
    if melhor is None or melhor["valor"] <= 0:
        return 0.0, np.zeros_like(ctx.cinza)
    return melhor["valor"], melhor["regiao"]


def candidato_furinho_tophat(ctx: Contexto, refs: Referencias, k: int = TOPHAT_FURINHO_K,
                             limiar: int = TOPHAT_FURINHO_LIMIAR, usar_envelope: bool = True,
                             circ_min: float = 0.0) -> tuple[float, np.ndarray]:
    """Método anterior do furinho (candidato): top-hat e black-hat de k px acima do envelope,
    limiar, fechamento + abertura; valor = área da maior mancha.

    Ele mede contraste local, e não forma: acusava também arranhões, impressões falhas e
    rachaduras, e tudo o que achava os outros especialistas também achavam. circ_min > 0 conta
    só as manchas com circularidade >= circ_min (candidato "circularidade sem o LoG": a máscara
    do top-hat sai picotada e mesmo os furos têm circularidade baixa)."""
    claro = acima_do_envelope(ctx, refs, "claro", k, usar_envelope)
    escuro = acima_do_envelope(ctx, refs, "escuro", k, usar_envelope)
    mascara = np.where((claro > limiar) | (escuro > limiar), 255, 0).astype(np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, _ee(FURINHO_FECHAMENTO))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, _ee(FURINHO_ABERTURA))
    if circ_min <= 0:
        return _maior_componente(mascara), mascara
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    maior = 0.0
    for i in range(1, n):
        x, y, w, h = stats[i, :4]
        mancha = np.where(rotulos[y:y + h, x:x + w] == i, 255, 0).astype(np.uint8)
        if circularidade(mancha)[1] >= circ_min:
            maior = max(maior, float(stats[i, cv2.CC_STAT_AREA]))
    return maior, mascara


# =============================================================================
# 5. ARRANHÃO: concentração de fragmentos finos e claros
# =============================================================================
def _fragmentos(ctx: Contexto, refs: Referencias, k: int, limiar: int,
                usar_envelope: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    claro = acima_do_envelope(ctx, refs, "claro", k, usar_envelope)
    binaria = np.where(claro > limiar, 255, 0).astype(np.uint8)
    n, rotulos, stats, _ = cv2.connectedComponentsWithStats(binaria, connectivity=8)
    return binaria, rotulos, stats


def especialista_arranhao(ctx: Contexto, refs: Referencias, k: int = ARRANHAO_K,
                          limiar: int = ARRANHAO_LIMIAR, usar_envelope: bool = True,
                          morfologia: bool = True) -> tuple[float, np.ndarray]:
    """Maior nº de pixels de fragmentos finos dentro de uma janela de 101x101 px.

    morfologia=False conta todos os fragmentos (sem remover os menores que ARRANHAO_AREA_MIN)
    e devolve a máscara sem o fechamento (só para medir o efeito da morfologia)."""
    _, rotulos, stats = _fragmentos(ctx, refs, k, limiar, usar_envelope)
    manter = stats[:, cv2.CC_STAT_AREA] >= (ARRANHAO_AREA_MIN if morfologia else 1)
    manter[0] = False
    frag = manter[rotulos]
    densidade = cv2.boxFilter(frag.astype(np.float32), -1, (ARRANHAO_JANELA, ARRANHAO_JANELA),
                              normalize=False)
    # máscara: os fragmentos, unidos por um fechamento (o feixe de riscos vira uma região)
    mascara = np.where(frag, 255, 0).astype(np.uint8)
    if morfologia:
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, _ee(15))
    return float(densidade.max()), mascara


def candidato_arranhao_maior_risco(ctx: Contexto, refs: Referencias,
                                   usar_envelope: bool = True) -> float:
    """Candidato descartado: comprimento (px) do maior fragmento. O risco sai picotado, então
    o maior pedaço diz menos que a concentração de pedaços."""
    _, _, stats = _fragmentos(ctx, refs, ARRANHAO_K, ARRANHAO_LIMIAR, usar_envelope)
    if len(stats) <= 1:
        return 0.0
    return float(np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT]).max())


def candidato_arranhao_canny(ctx: Contexto) -> float:
    """Candidato descartado (bordas da Etapa 1): nº de pixels de borda do Canny na zona válida."""
    from pipeline import detectar_bordas_canny
    return float(np.count_nonzero(detectar_bordas_canny(ctx.cinza)[ctx.valida]))


# =============================================================================
# CASCATA: os 5 especialistas em sequência
# =============================================================================
ESPECIALISTAS = {
    "squeeze": especialista_squeeze,
    "faulty_imprint": especialista_impressao,
    "crack": especialista_rachadura,
    "poke": especialista_furinho,
    "scratch": especialista_arranhao,
}


def examinar(bgr: np.ndarray, refs: Referencias,
             ctx: Contexto | None = None) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Passa a imagem pelos 5 especialistas. Devolve {estágio: valor} e {estágio: máscara}.
    `ctx` evita refazer preparar(bgr) quando quem chama já tem o contexto."""
    if ctx is None:
        ctx = preparar(bgr)
    valores, mascaras = {}, {}
    for nome in ESTAGIOS:
        valores[nome], mascaras[nome] = ESPECIALISTAS[nome](ctx, refs)
    return valores, mascaras


def referencias_do_banco() -> Referencias:
    from particao import carregar_particao
    return construir_referencias([l["caminho"] for l in carregar_particao("banco")])


# =============================================================================
# Passo a passo de uma imagem (figura)
# =============================================================================
def figura_passo_a_passo(bgr: np.ndarray, gt: np.ndarray, refs: Referencias, titulo: str,
                         limiares: dict[str, float] | None = None,
                         salvar: str | Path | None = None, mostrar: bool = True) -> None:
    """Linha de cima: o que cada especialista olha. Linha de baixo: o que ele acusa."""
    import matplotlib.pyplot as plt

    ctx = preparar(bgr)
    ys, xs = np.where(ctx.capsula > 0)
    y0, y1 = max(ys.min() - 40, 0), min(ys.max() + 40, bgr.shape[0])
    x0, x1 = max(xs.min() - 40, 0), min(xs.max() + 40, bgr.shape[1])
    rec = lambda im: im[y0:y1, x0:x1]

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mapa_log, blobs = blobs_furinho(ctx, refs)
    realce = np.clip(mapa_log * 6, 0, 255).astype(np.uint8)
    fino = np.clip(acima_do_envelope(ctx, refs, "claro", ARRANHAO_K).astype(int) * 25, 0, 255)
    zonas = (rgb * 0.45).astype(np.uint8)
    zonas[ctx.valida] = rgb[ctx.valida]
    entradas = {
        "squeeze": (ctx.capsula, "silhueta (Otsu)", "gray"),
        "faulty_imprint": (ctx.impressao, '"500" (histograma do realce)', "gray"),
        "crack": (ctx.preto, "tampa preta (Otsu no a*)", "gray"),
        "poke": (realce, "LoG acima do envelope (círculo = blob do valor)", "magma"),
        "scratch": (fino, f"top-hat {ARRANHAO_K} px acima do envelope", "gray"),
    }

    fig, eixos = plt.subplots(2, 6, figsize=(24, 6.2))
    for ax in eixos.ravel():
        ax.axis("off")
    eixos[0, 0].imshow(rec(rgb))
    eixos[0, 0].set_title("imagem (verde = defeito real)", fontsize=10)
    if gt.any():
        eixos[0, 0].contour(rec(gt) > 0, levels=[0.5], colors="lime", linewidths=1)
    eixos[1, 0].imshow(rec(zonas))
    eixos[1, 0].set_title('zona válida dos especialistas 4 e 5\n(sem contorno, "500", logo e divisa)',
                          fontsize=10)

    acusou, primeira = [], None
    for col, nome in enumerate(ESTAGIOS, start=1):
        valor, mascara = ESPECIALISTAS[nome](ctx, refs)
        img, legenda, cmap = entradas[nome]
        eixos[0, col].imshow(rec(img), cmap=cmap)
        eixos[0, col].set_title(f"{col}. {NOMES[nome]}\n{legenda}", fontsize=10)
        if nome == "poke" and valor > 0:
            b = max(blobs, key=lambda b: b["valor"])
            eixos[0, col].add_patch(plt.Circle((b["x"] - x0, b["y"] - y0), b["raio"], fill=False,
                                               color="cyan", linewidth=1))
        sobre = (rgb * 0.55).astype(np.uint8)
        sobre[mascara > 0] = (255, 60, 60)
        eixos[1, col].imshow(rec(sobre))
        texto = f"valor = {valor:.3f}" if valor < 1 else (
            f"valor = {valor:.1f}" if valor < 100 else f"valor = {valor:.0f}")
        if limiares is not None:
            disparou = valor > limiares[nome]
            texto += f"  (limiar {limiares[nome]:.3g})\n" + ("DEFEITO" if disparou else "passou")
            if disparou:
                acusou.append(NOMES[nome])
                primeira = primeira or nome
        eixos[1, col].set_title(texto, fontsize=10,
                                color="crimson" if "DEFEITO" in texto else "black")
        if gt.any():
            eixos[1, col].contour(rec(gt) > 0, levels=[0.5], colors="lime", linewidths=1)
    if limiares is not None and acusou:
        # tipo = primeira etapa que acusa; etapa 2 com diferença clara no "500" = furinho
        if (primeira == "faulty_imprint" and diferenca_do_500(ctx, refs)[CHAVE_FRACAO_CLARA]
                > limiares.get(CHAVE_FRACAO_CLARA, 1.0)):
            primeira = "poke"
        titulo += (f"   ->   DEFEITUOSA, tipo {NOMES[primeira]}   (acusaram: "
                   + ", ".join(acusou) + ")")
    elif limiares is not None:
        titulo += "   ->   BOA"
    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    if salvar:
        fig.savefig(salvar, dpi=80)
    if mostrar:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Especialistas por defeito: passo a passo")
    parser.add_argument("--defeito", default="crack",
                        choices=["good", "crack", "faulty_imprint", "poke", "scratch", "squeeze"])
    parser.add_argument("--indice", type=int, default=0)
    parser.add_argument("--salvar", default=None, help="arquivo .png de saída")
    args = parser.parse_args()

    if args.salvar:
        import matplotlib
        matplotlib.use("Agg")
    print("Montando as referências com as 30 boas do banco...")
    refs = referencias_do_banco()
    bgr = carregar_imagem(caminho_teste(args.defeito, args.indice))
    gt = carregar_mascara_gt(args.defeito, args.indice, bgr.shape[:2])
    # Limiares finais gravados por cascata.py (se já foi rodado)
    limiares = None
    arq_limiares = RAIZ / "dados" / "limiares_cascata.json"
    if arq_limiares.exists():
        import json
        limiares = next(iter(json.loads(arq_limiares.read_text(encoding="utf-8")).values()))
    valores, _ = examinar(bgr, refs)
    for nome in ESTAGIOS:
        situacao = ""
        if limiares:
            situacao = "  ACUSOU" if valores[nome] > limiares[nome] else "  passou"
            situacao = f"  (limiar {limiares[nome]:.3g}){situacao}"
        print(f"  {NOMES[nome]:<16} {valores[nome]:10.3f}{situacao}")
    figura_passo_a_passo(bgr, gt, refs, f"capsule / {args.defeito} / {args.indice:03d}",
                         limiares=limiares, salvar=args.salvar, mostrar=not args.salvar)


if __name__ == "__main__":
    main()
