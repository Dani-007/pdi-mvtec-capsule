"""
Etapa 7 - Qual método é o ideal para cada defeito? Comparação dos candidatos.

Para cada defeito foram testados alguns métodos. Cada um dá um valor por imagem, e a comparação
usa duas medidas, sempre do defeito-alvo contra as cápsulas boas de 'classificacao':

    AUC     chance de uma cápsula com o defeito ter valor maior que uma boa (0,50 = acaso);
    achadas fração das cápsulas com o defeito que o método acha quando o limiar é o percentil
            98 das boas de treino (validação cruzada com os folds da partição), ou seja,
            aceitando ~2% de alarme falso.

O escolhido de cada defeito (marcado com *) é o que está em especialistas.py. A primeira
linha de cada bloco é o método único das Etapas 3-4 (área da maior mancha suspeita do golden
template), lido de dados/atributos.csv, para referência.

Saída: resultados/etapa7/tabela_candidatos.txt|csv
Uso:   python src/comparar_candidatos.py      (~2 min)
"""

from __future__ import annotations

import csv

import numpy as np

import cascata
from cascata import DIR_RESULT, TIPOS, auc, mapear
from especialistas import (
    FURINHO_K, NOMES, _centroide, candidato_arranhao_canny, candidato_arranhao_maior_risco,
    candidato_impressao_area, candidato_squeeze_descritores, construir_referencias,
    especialista_arranhao, especialista_furinho, especialista_impressao, especialista_rachadura,
    especialista_squeeze, preparar, segmentar_laranja,
)
from particao import carregar_particao
from pipeline import RAIZ, carregar_imagem

K_GRANDE = 25
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
    ("crack", "cr_a", "forma da tampa preta, Otsu no a* (Lab)", True),
    ("crack", "fu_env15", "contraste local 15 px (o especialista do furinho)", False),
    ("poke", "fu_sem_env", "top-hat/black-hat 15 px, sem envelope", False),
    ("poke", "fu_env25", "top-hat/black-hat 25 px, acima do envelope", False),
    ("poke", "fu_env15", "top-hat/black-hat 15 px, acima do envelope", True),
    ("scratch", "ar_canny", "pixels de borda do Canny (Etapa 1)", False),
    ("scratch", "ar_maior_sem_env", "maior risco (top-hat 9 px), sem envelope", False),
    ("scratch", "ar_maior", "maior risco (top-hat 9 px), acima do envelope", False),
    ("scratch", "ar_dens_sem_env", "concentração de fragmentos, sem envelope", False),
    ("scratch", "ar_dens", "concentração de fragmentos, acima do envelope", True),
]

_TAMPAS_CINZA = None   # tampas pretas do banco segmentadas pelo cinza (candidato cr_cinza)


def _tampa_cinza(ctx) -> np.ndarray:
    laranja = segmentar_laranja(ctx.bgr, ctx.capsula, canal="cinza")
    return np.where((ctx.capsula > 0) & (laranja == 0), 255, 0).astype(np.uint8)


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
    v["cr_cinza"] = especialista_rachadura(ctx, refs, tampa=_tampa_cinza(ctx),
                                           tampas_ref=_TAMPAS_CINZA)[0]
    v["cr_a"] = especialista_rachadura(ctx, refs)[0]
    v["fu_sem_env"] = especialista_furinho(ctx, refs, usar_envelope=False)[0]
    v["fu_env25"] = especialista_furinho(ctx, refs, k=K_GRANDE)[0]
    v["fu_env15"] = especialista_furinho(ctx, refs)[0]
    v["ar_canny"] = candidato_arranhao_canny(ctx)
    v["ar_maior_sem_env"] = candidato_arranhao_maior_risco(ctx, refs, usar_envelope=False)
    v["ar_maior"] = candidato_arranhao_maior_risco(ctx, refs)
    v["ar_dens_sem_env"] = especialista_arranhao(ctx, refs, usar_envelope=False)[0]
    v["ar_dens"] = especialista_arranhao(ctx, refs)[0]
    return v


def achadas_cv(valor: np.ndarray, alvo: np.ndarray, boas: np.ndarray, treino_extra: np.ndarray,
               fold: np.ndarray) -> float:
    """Fração do alvo acima do percentil 98 das boas de treino, decidida fold a fold."""
    achou = np.zeros(len(valor), bool)
    for k in range(5):
        treino = treino_extra | (boas & (fold != k))
        limiar = np.percentile(valor[treino], 100 * (1 - ALFA_COMPARACAO))
        achou[fold == k] = valor[fold == k] > limiar
    return float(achou[alvo].mean())


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
    global _TAMPAS_CINZA
    DIR_RESULT.mkdir(parents=True, exist_ok=True)
    banco = [l["caminho"] for l in carregar_particao("banco")]
    print("Montando as referências com as 30 boas do banco...")
    refs = construir_referencias(banco, ks_claro=(9, FURINHO_K, K_GRANDE),
                                 ks_escuro=(FURINHO_K, K_GRANDE))
    _TAMPAS_CINZA = []
    for c in banco:
        t = _tampa_cinza(preparar(carregar_imagem(c)))
        _TAMPAS_CINZA.append((t, _centroide(t)))

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
    texto = "\n".join(txt)
    (DIR_RESULT / "tabela_candidatos.txt").write_text(texto, encoding="utf-8")
    with open(DIR_RESULT / "tabela_candidatos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tabela[0].keys()), lineterminator="\n")
        w.writeheader()
        w.writerows(tabela)
    print("\n" + texto)
    print(f"Tabelas em {DIR_RESULT}")


if __name__ == "__main__":
    main()
