"""Etapa 2: partição própria do dataset.

    banco          30 boas de train/good  -> referências
    calibracao     20 boas de train/good  -> limiares (só a variação normal)
    classificacao  192 boas + 109 defeituosas -> avaliação, 5 folds estratificados por tipo

Uso:  python src/particao.py   -> dados/particao.csv
"""

from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path

from pipeline import DATASET_DIR, GOLDEN_N_BANCO

N_CALIBRACAO = 20
N_FOLDS = 5
SEMENTE = 42
TIPOS_DEFEITO = ["crack", "faulty_imprint", "poke", "scratch", "squeeze"]
SAIDA = Path(__file__).resolve().parent.parent / "dados" / "particao.csv"


def montar_particao() -> list[dict]:
    linhas = []

    def adicionar(caminho: Path, tipo: str, conjunto: str) -> None:
        linhas.append({
            "arquivo": caminho.relative_to(DATASET_DIR).as_posix(),
            "tipo": tipo,
            "rotulo": "normal" if tipo == "good" else "defeituosa",
            "conjunto": conjunto,
            "fold": "",
        })

    bons_treino = sorted((DATASET_DIR / "train" / "good").glob("*.png"))
    fim_cal = GOLDEN_N_BANCO + N_CALIBRACAO
    for p in bons_treino[:GOLDEN_N_BANCO]:
        adicionar(p, "good", "banco")
    for p in bons_treino[GOLDEN_N_BANCO:fim_cal]:
        adicionar(p, "good", "calibracao")
    for p in bons_treino[fim_cal:]:
        adicionar(p, "good", "classificacao")
    for tipo in ["good"] + TIPOS_DEFEITO:
        for p in sorted((DATASET_DIR / "test" / tipo).glob("*.png")):
            adicionar(p, tipo, "classificacao")

    # Folds estratificados por tipo: embaralha cada tipo e distribui em rodízio.
    rng = random.Random(SEMENTE)
    for tipo in ["good"] + TIPOS_DEFEITO:
        grupo = [l for l in linhas if l["conjunto"] == "classificacao" and l["tipo"] == tipo]
        rng.shuffle(grupo)
        for i, linha in enumerate(grupo):
            linha["fold"] = i % N_FOLDS
    return linhas


def carregar_particao(conjunto: str | None = None) -> list[dict]:
    """Lê dados/particao.csv (opcionalmente filtrando por conjunto)."""
    with open(SAIDA, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    for l in linhas:
        l["fold"] = int(l["fold"]) if l["fold"] != "" else None
        l["caminho"] = DATASET_DIR / l["arquivo"]
    return [l for l in linhas if conjunto is None or l["conjunto"] == conjunto]


def main() -> None:
    linhas = montar_particao()
    SAIDA.parent.mkdir(exist_ok=True)
    with open(SAIDA, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arquivo", "tipo", "rotulo", "conjunto", "fold"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(linhas)

    print(f"Partição salva em {SAIDA} ({len(linhas)} imagens)\n")
    for conjunto, n in Counter(l["conjunto"] for l in linhas).items():
        print(f"  {conjunto:<14} {n:4d}")
    print("\nConjunto 'classificacao' por tipo e fold:")
    cls = [l for l in linhas if l["conjunto"] == "classificacao"]
    print(f"  {'tipo':<15}" + "".join(f"  f{k}" for k in range(N_FOLDS)) + "  total")
    for tipo in ["good"] + TIPOS_DEFEITO:
        cont = Counter(l["fold"] for l in cls if l["tipo"] == tipo)
        print(f"  {tipo:<15}" + "".join(f"{cont[k]:4d}" for k in range(N_FOLDS))
              + f"  {sum(cont.values()):5d}")


if __name__ == "__main__":
    main()
