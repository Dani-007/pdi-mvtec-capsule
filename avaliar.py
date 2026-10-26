"""
Avaliação quantitativa sobre todo o test/ de capsule.

Gera as tabelas usadas para justificar os parâmetros de pipeline.py:
    1. Quanto do GT sobrevive a cada erosão da ROI.
    2. Bordas (Canny/Sobel) x tamanho da abertura: falsos positivos vs. detecção.
    3. Golden template x limiar x abertura: Dice/precisão por tipo de defeito.

Métricas:
    hit      fração de imagens defeituosas com >= 30 px marcados dentro do GT
    FP_boa   média de pixels marcados por imagem boa (todo pixel é falso positivo)
    FP_def   média de pixels marcados FORA do GT nas imagens defeituosas
    prec     precisão pixel a pixel global (soma TP / soma marcados) nas defeituosas

Uso:  python avaliar.py   (~3 min; o golden template domina o tempo)
"""

import numpy as np
import cv2

from pipeline import (
    DATASET_DIR, GOLDEN_N_BANCO, MORF_FECHAMENTO_PREVIO,
    aplicar_abertura, carregar_banco_referencias, carregar_imagem, carregar_mascara_gt,
    converter_cinza, detectar_bordas_canny, detectar_bordas_sobel, filtro_bilateral,
    golden_template, metricas_pixel, segmentar_capsula, selecionar_referencia,
)

TIPOS = ["crack", "faulty_imprint", "poke", "scratch", "squeeze", "good"]
MIN_PX_HIT = 30


def carregar_teste():
    dados = []
    for tipo in TIPOS:
        for p in sorted((DATASET_DIR / "test" / tipo).glob("*.png")):
            img = carregar_imagem(p)
            suave = filtro_bilateral(converter_cinza(img))
            mascara, _ = segmentar_capsula(suave, erosao=0)
            gt = carregar_mascara_gt(tipo, int(p.stem), img.shape[:2])
            dados.append({"tipo": tipo, "suave": suave, "mascara": mascara, "gt": gt})
    return dados


def resumir(nome: str, registros: list[tuple[str, dict]]) -> None:
    defeito = [m for t, m in registros if t != "good"]
    boas = [m for t, m in registros if t == "good"]
    tp = sum(m["tp"] for m in defeito)
    fp = sum(m["fp"] for m in defeito)
    dice_tipo = {t: np.mean([m["dice"] for tt, m in registros if tt == t]) for t in TIPOS[:-1]}
    print(f"{nome:<28} hit={np.mean([m['tp'] >= MIN_PX_HIT for m in defeito]):.2f}  "
          f"FP_boa={np.mean([m['fp'] for m in boas]):7.0f}  "
          f"FP_def={np.mean([m['fp'] for m in defeito]):7.0f}  prec={tp / (tp + fp + 1e-9):.3f}  "
          + " ".join(f"{t[:6]}={d:.2f}" for t, d in dice_tipo.items()))


def avaliar_roi(dados) -> None:
    print("\n=== 1. Fração do GT dentro da ROI x erosão ===")
    defeituosas = [d for d in dados if d["tipo"] != "good"]
    for er in (0, 5, 9, 15, 21, 31):
        cobertura = []
        for d in defeituosas:
            roi = d["mascara"] if er == 0 else cv2.erode(
                d["mascara"], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (er, er)))
            cobertura.append(np.sum((d["gt"] > 0) & (roi > 0)) / max(np.sum(d["gt"] > 0), 1))
        print(f"erosão {er:2d}px: média={np.mean(cobertura):.3f}  mínimo={np.min(cobertura):.3f}")


def avaliar_bordas(dados) -> None:
    print("\n=== 2. Bordas + abertura (dice por tipo é baixo por natureza: borda x região) ===")
    configs = {
        "Canny 25/60": lambda s, r: (detectar_bordas_canny(s, 25, 60, roi=r), MORF_FECHAMENTO_PREVIO),
        "Canny 15/40": lambda s, r: (detectar_bordas_canny(s, 15, 40, roi=r), MORF_FECHAMENTO_PREVIO),
        "Canny 10/25": lambda s, r: (detectar_bordas_canny(s, 10, 25, roi=r), MORF_FECHAMENTO_PREVIO),
        "Sobel |G|>40": lambda s, r: (detectar_bordas_sobel(s, limiar=40, roi=r)[1], 0),
        "Sobel |G|>60": lambda s, r: (detectar_bordas_sobel(s, limiar=60, roi=r)[1], 0),
    }
    registros = {}
    for d in dados:
        _, roi = segmentar_capsula(d["suave"])
        for nome, fn in configs.items():
            bordas, fech = fn(d["suave"], roi)
            saidas = {0: bordas, **aplicar_abertura(bordas, fechamento_previo=fech)}
            for k, m in saidas.items():
                registros.setdefault(f"{nome} | abertura {k}", []).append(
                    (d["tipo"], metricas_pixel(m, d["gt"])))
    for nome, regs in registros.items():
        resumir(nome, regs)


def avaliar_golden(dados) -> None:
    print("\n=== 3. Golden template (referência mais próxima) ===")
    bons = sorted((DATASET_DIR / "train" / "good").glob("*.png"))
    banco = carregar_banco_referencias(bons[:GOLDEN_N_BANCO])
    registros = {}
    for d in dados:
        ref, alinhada = selecionar_referencia(d["suave"], banco)
        for limiar in (20, 25, 30, 40):
            for ab in (0, 3, 5):
                _, m = golden_template(alinhada, ref["imagem"], limiar=limiar,
                                       abertura=ab, roi=ref["mascara"])
                registros.setdefault(f"limiar {limiar} | abertura {ab}", []).append(
                    (d["tipo"], metricas_pixel(m, d["gt"])))
    for nome, regs in registros.items():
        resumir(nome, regs)


if __name__ == "__main__":
    dados = carregar_teste()
    print(f"{len(dados)} imagens de teste carregadas")
    avaliar_roi(dados)
    avaliar_bordas(dados)
    avaliar_golden(dados)
