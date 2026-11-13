# Projeto 4 — Detecção e classificação de defeitos industriais (MVTec AD · *capsule*)

Disciplina de Processamento Digital de Imagens — UFSCar · Prof. Dr. Ricardo J. Ferrari

> **Questão de projeto:** é possível detectar e classificar defeitos usando apenas processamento
> clássico e atributos interpretáveis?

## Sumário

1. [Contexto e objetivo](#1-contexto-e-objetivo)
2. [Dados e escopo](#2-dados-e-escopo)
3. [Andamento](#3-andamento)
4. [Pipeline (Etapas 1–4)](#4-pipeline-etapas-14)
5. [Avaliação](#5-avaliação)
6. [Resultados](#6-resultados)
7. [Discussão e limitações](#7-discussão-e-limitações)
8. [Próximos passos](#8-próximos-passos)
9. [Como executar](#9-como-executar)
10. [Estrutura do repositório](#10-estrutura-do-repositório)

---

## 1. Contexto e objetivo

O projeto simula a **inspeção visual automática** de uma linha de produção. A partir da imagem de uma
cápsula, o sistema deve:

1. **localizar** regiões anômalas (onde está o defeito);
2. **descrever** essas regiões por atributos clássicos (forma, intensidade, textura);
3. **classificar** a amostra como normal/defeituosa e, quando possível, pelo tipo de defeito.

Por restrição do projeto, são usadas apenas **técnicas clássicas de PDI** e **atributos
interpretáveis**, sem deep learning. O foco é medir até onde essa abordagem chega e identificar quais
etapas mais limitam o desempenho.

```
Denoising → Bordas → Morfologia → Segmentação → Atributos → Classificação
└──────────── Etapa 1: localização ────────────┘ └────── decisão ──────────┘
```

## 2. Dados e escopo

- **Base:** [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), categoria
  **capsule** (licença CC BY-NC-SA 4.0). Imagens RGB de 1000×1000 px com uma cápsula preta/laranja
  com a impressão "500" sobre fundo claro.

| Conjunto | Conteúdo |
|---|---|
| `train/good` | 219 cápsulas normais |
| `test/good` | 23 cápsulas normais |
| `test/<defeito>` | 109 defeituosas: *crack*, *faulty_imprint*, *poke*, *scratch*, *squeeze* |
| `ground_truth/` | máscara binária de cada defeito, usada **só na avaliação** |

- **Medições iniciais** (tons de cinza, após o bilateral), que orientaram os parâmetros:

| Região | Intensidade |
|---|---|
| fundo | ≈ 200 |
| metade laranja | ≈ 85–95 |
| metade preta | ≈ 20–40 |
| impressão "500" | ≈ 86 → 121 (baixo contraste) |

> Uso didático: os resultados não devem ser lidos como benchmark oficial do MVTec AD.

## 3. Andamento

| Etapa | Descrição | Situação |
|---|---|---|
| **1** | Localização do defeito: duas estratégias comparadas | ✅ concluída |
| **2** | Partição própria (`src/particao.py`) | ✅ concluída |
| **3** | Máscara final: golden template + morfologia completa (`src/segmentacao.py`) | ✅ concluída |
| **4** | Atributos: ficha de cada cápsula, com a impressão "500" segmentada pelo histograma (`src/impressao.py`, `src/atributos.py`) | ✅ concluída |
| **5–6** | Classificação e resultados finais | ⏳ próximos passos |

## 4. Pipeline (Etapas 1–4)

### 4.1 Pré-processamento
| Passo | Técnica | Justificativa |
|---|---|---|
| Conversão | BGR → tons de cinza | Os defeitos aparecem como variação de intensidade; reduz para 1 canal. Ressalva: na metade preta o contraste é baixo. |
| Denoising | **Filtro bilateral** (d = 9, σ_cor = 40, σ_espaço = 9) | Suaviza o ruído **preservando bordas**: vizinhos de intensidade muito diferente recebem peso ≈ 0. O Gaussiano espalharia o degrau dos defeitos finos e a mediana tende a apagar estruturas finas. σ_cor = 40 fica entre o ruído (5–15 níveis) e as bordas de defeito (≥ 30). |

### 4.2 Região de interesse (ROI)
1. **Limiarização de Otsu invertida:** a cápsula é mais escura que o fundo, e o limiar (≈ 130) é
   escolhido automaticamente no vale do histograma.
2. **Fechamento 15×15:** tapa os buracos deixados pela impressão clara e pelos reflexos.
3. **Maior componente conexo**, preenchido: descarta sujeiras no fundo.
4. **Erosão de 9 px:** remove o contorno da cápsula, que gera falso positivo por ter alto contraste.
   Com 9 px, ~93% da área do GT continua dentro da ROI.

### 4.3 Estratégia A — bordas + morfologia
- **Canny** (limiares 15/40, `L2gradient`): histerese com razão ≈ 1:2,7. O limiar alto de 40
  inicia ~1–2% dos pixels (p99 do gradiente na cápsula ≈ 60); o baixo de 15 dá continuidade a bordas
  fracas, como arranhões.
- **Sobel** 3×3, magnitude com limiar fixo **60** (≈ p99 dentro da cápsula).
- **Abertura** com elemento elíptico de 3, 5, 7 e 9 px, para medir o efeito da morfologia sobre os
  falsos positivos. No Canny, um **fechamento de 5 px** é aplicado antes, porque bordas de 1 px
  seriam apagadas por qualquer abertura.

### 4.4 Estratégia B — golden template
1. **Banco de referência:** 30 cápsulas de `train/good`.
2. **Referência mais parecida:** o teste é alinhado a cada referência, e fica a de menor diferença
   média dentro da ROI. Motivo: a cápsula **gira em torno do próprio eixo** entre as fotos, e a
   posição do "500" muda. Uma referência fixa ou a média das boas teria a impressão em outro lugar
   ou borrada.
3. **Alinhamento** por ECC, modelo euclidiano (rotação + translação), estimado em ¼ da resolução.
4. **Diferença absoluta** |teste − referência|: estruturas presentes nas duas imagens se cancelam.
5. **Limiar 30:** ≈ **percentil 99** da diferença entre cápsulas **boas** e suas referências,
   calibrado em 20 imagens de `train/good` fora do banco.
6. **Abertura 5×5:** remove fragmentos de 1–3 px causados pelo erro residual de alinhamento, mais
   finos que os defeitos.

### 4.5 Etapa 2 — partição própria (`src/particao.py` → `dados/particao.csv`)
Cada imagem recebe um papel fixo, para que nenhum parâmetro seja escolhido olhando as imagens que
dão a nota final:

| Conjunto | Imagens | Uso |
|---|---|---|
| `banco` | 30 normais (`train/good` 000–029) | referências do golden template (as mesmas da Etapa 1) |
| `calibracao` | 20 normais (`train/good` 030–049) | calibrar limiares e tamanhos usando só a variação normal |
| `classificacao` | 169 + 23 normais e 109 defeituosas | avaliação da segmentação; atributos e classificação com validação cruzada em **5 folds** estratificados por tipo (semente 42) |

### 4.6 Etapa 3 — máscara final (`src/segmentacao.py`)
Golden template da Etapa 1 seguido de morfologia completa, aplicada de forma cumulativa para medir o
efeito de cada operação:

| Passo | Operação | Objetivo |
|---|---|---|
| C0 | \|teste − ref\| > T, com T = p99 da diferença nas 20 boas de calibração (= 32) | marcação inicial |
| C1 | abertura 5×5 | remover fragmentos finos (= Etapa 1) |
| C2 | ROI erodida 9 px | excluir o contorno da cápsula |
| C3 | fechamento 5×5 | unir pedaços do mesmo defeito |
| C4 | preenchimento de buracos (*flood fill*) | manchas sólidas |
| C5 | remoção de componentes conexos < A_MIN, com A_MIN = p95 da maior mancha nas boas de calibração (= 3.671 px) | remover manchas pequenas |

A máscara **C4** de cada imagem de `classificacao` é salva em `dados/gerado/mascaras/` para a etapa
de atributos (ver resultados na seção 6).

### 4.7 Etapa 4 — atributos (`src/impressao.py`, `src/atributos.py` → `dados/atributos.csv`)

**Impressão "500" pelo histograma (orientação do professor).** Dentro da máscara da cápsula, o
histograma separa metade preta (< 45), logo (45–75), metade laranja (75–112) e impressão (> 112).
Como a iluminação deixa a metade laranja com dois tons, o limiar é calculado no histograma do
**realce local** (cinza − mediana 45×45), só na metade clara e longe do contorno, por Otsu; depois,
abertura 5×5 e componentes conexos com forma de algarismo (`resultados/etapa4/impressao_500.png`).
O "500" **não é apagado** da máscara do defeito, porque os defeitos também caem sobre ele (impressão
falha ≈ 89% da área, furinho e arranhão ≈ 45%): ele vira atributos.

**Ficha (28 atributos por imagem, 301 imagens de `classificacao`)**, em 3 grupos para a comparação
do enunciado:

| Grupo | Atributos | Calculados sobre |
|---|---|---|
| Forma | nº de manchas, área total, área, perímetro, compacidade (P²/4πA), excentricidade, extensão, solidez, momentos de Hu 1–3 | maior mancha da máscara C4 |
| Textura | média e desvio do cinza, média e máximo da diferença ao modelo, suavidade, assimetria, uniformidade e entropia do histograma, entropia e uniformidade do LBP (8 vizinhos, raio 1) | pixels da maior mancha |
| Posição e impressão | fração da mancha sobre o "500", fração na metade preta, distância ao contorno; área relativa do "500" (÷ mediana nas 20 boas de calibração), nº de componentes, nº de buracos e variação de área dos algarismos | mancha e impressão |

Cápsula sem mancha: atributos da mancha = 0 (e `n_manchas` = 0). Nenhum atributo usa os rótulos.
Distribuições por tipo em `resultados/etapa4/atributos_por_tipo.png`: a área relativa e o nº de
buracos do "500" destacam a impressão falha; área, compacidade, diferença ao modelo e distância ao
contorno destacam o *squeeze*.

## 5. Avaliação

Pixel a pixel contra o ground truth. Etapa 1: as 132 imagens de `test/` (109 defeituosas + 23
boas). Etapa 3: as 301 imagens do conjunto `classificacao` (109 defeituosas + 192 boas).

| Métrica | Significado |
|---|---|
| **Precisão** = TP/(TP+FP) | Fração do que foi marcado que é defeito: mede o alarme falso |
| **Recall** = TP/(TP+FN) | Fração do defeito real encontrada |
| **Dice** / **IoU** | Sobreposição entre a máscara prevista e o GT |
| **hit** | Fração de imagens defeituosas com ≥ 30 px marcados dentro do GT |
| **FP_boa** | Média de pixels marcados por imagem **normal**: todos são falsos positivos |

Para mapas de borda, o Dice é baixo por natureza (a borda tem 1–3 px e o GT é a região inteira).
Nesse caso, o indicador principal é o número de falsos positivos.

## 6. Resultados

Resultados completos em `resultados/etapa1/avaliacao.txt`; figuras de exemplo em `resultados/etapa1/*.png`.

**Estratégia A — efeito da abertura (Canny 15/40):**

| Abertura | hit | FP_boa | Precisão |
|---|---:|---:|---:|
| — | 0,72 | 2.476 | 0,09 |
| 3 px | 0,42 | 247 | 0,24 |
| 5 px | 0,22 | 114 | 0,25 |
| 9 px | 0,02 | 0 | 0,70 |

**Estratégia B — efeito do limiar (abertura 5):**

| Limiar | hit | FP_boa | Precisão | Dice crack / faulty / poke / scratch / squeeze |
|---|---:|---:|---:|---|
| 20 | 0,83 | 6.168 | 0,25 | 0,21 / 0,17 / 0,20 / 0,01 / 0,27 |
| 25 | 0,72 | 4.177 | 0,27 | 0,21 / 0,12 / 0,18 / 0,01 / 0,25 |
| **30** | **0,63** | **2.792** | **0,31** | **0,21 / 0,07 / 0,16 / 0,00 / 0,25** |
| 40 | 0,42 | 1.029 | 0,42 | 0,21 / 0,00 / 0,12 / 0,00 / 0,24 |

**Etapa 3 — efeito da morfologia** (`resultados/etapa3/`). Golden template com parâmetros
calibrados só nas 20 normais de `calibracao` (T = 32 = p99 da diferença entre boas; A_MIN = 3.671 px
= p95 da maior mancha que sobra em cada boa), avaliado nas 301 imagens de `classificacao`:

| Etapa (cumulativa) | hit | FP px/boa | Boas com alarme | Precisão | Dice |
|---|---:|---:|---:|---:|---:|
| C0 limiar | 0,64 | 2.844 | 98% | 0,30 | 0,128 |
| C1 + abertura 5 (Etapa 1) | 0,57 | 2.381 | 85% | 0,33 | 0,129 |
| C2 + ROI erodida 9 px | 0,57 | 2.351 | 85% | 0,33 | 0,125 |
| C3 + fechamento 5 | 0,57 | 2.376 | 85% | 0,33 | 0,125 |
| C4 + preenchimento | 0,57 | 2.398 | 85% | 0,33 | 0,126 |
| C5 + remoção de componentes < A_MIN | 0,11 | 373 | 9% | 0,57 | 0,054 |

- A abertura é a única operação que reduz falsos positivos sem custo adicional relevante; ROI
  erodida, fechamento e preenchimento mudam pouco os números.
- A remoção por área elimina quase todos os alarmes nas boas, mas também quase todos os defeitos:
  as manchas falsas das cápsulas boas (impressão "500" deslocada) têm o **mesmo tamanho** dos
  defeitos, então a área sozinha não separa as duas coisas.
- Por isso a máscara levada para a Etapa 4 é a **C4**; separar "mancha falsa" de "defeito" fica a
  cargo dos atributos e do classificador. (Essa escolha foi feita depois de ver a tabela e é
  declarada aqui.)

## 7. Discussão e limitações

- **Bordas não separam defeito de estrutura normal.** A cápsula sem defeito já tem muitas bordas
  (impressão, logo, divisão das metades, contorno). A abertura que remove essas bordas remove os
  defeitos na mesma proporção, porque rachaduras e arranhões também são estruturas finas.
- **O golden template é a melhor estratégia**: maior precisão e Dice, e marca a região do defeito,
  não só o contorno. Mas ele acusa **qualquer** diferença, inclusive a variação normal da posição
  do "500".
- **Por tipo:** *crack*, *poke* e *squeeze* são localizados parcialmente. *faulty_imprint* fica
  mascarado pela variação normal da impressão. *scratch* é fino e de baixo contraste demais
  (Dice ≈ 0).
- **Trade-off do limiar:** reduzir o limiar aumenta a detecção à custa de mais falsos positivos.
- **Morfologia (Etapa 3):** só a abertura reduz falsos positivos de forma útil. Filtrar por área não
  funciona, porque as manchas falsas da impressão têm o tamanho dos defeitos. A ROI erodida também
  corta parte de defeitos que ficam na borda da cápsula (*squeeze*).

**Limitações metodológicas:**
1. Na Etapa 1, os parâmetros das bordas e as tabelas de sensibilidade foram avaliados no próprio
   `test/`. A partir da Etapa 2 isso foi corrigido com a partição própria; a única escolha feita
   depois de ver resultados (usar C4 em vez de C5) está declarada na seção 6.
2. Uma única referência por imagem não tolera a variação normal entre cápsulas boas.

## 8. Próximos passos

1. ~~Partição própria~~ (Etapa 2, feita).
2. ~~Morfologia completa, quantificando o efeito sobre os falsos positivos~~ (Etapa 3, feita).
3. **Localização:** tornar a referência tolerante à variação normal entre cápsulas boas.
4. ~~Atributos da região segmentada~~ (Etapa 4, feita; textura por estatísticas do histograma e
   LBP, sem GLCM, que não foi vista em aula).
5. **Classificação** normal/defeituoso e por tipo, com **SVM e Random Forest**, comparando atributos
   geométricos, de textura e a combinação completa.
6. **Métricas finais:**
   - F1, balanced accuracy e ROC-AUC;
   - resultados por tipo de defeito;
   - exemplos de falsos positivos e falsos negativos;
   - discussão das etapas que mais limitam o desempenho.

## 9. Como executar

1. Baixe a categoria **capsule** do MVTec AD e extraia em `capsule/`, na raiz do repositório.
2. Instale as dependências:
   ```
   pip install -r requirements.txt     # opencv-python, numpy, matplotlib
   ```
3. Rode os scripts **a partir da raiz do repositório**:
   ```
   # Etapa 1
   python src/pipeline.py --defeito crack --indice 0     # passo a passo de uma imagem
   python src/pipeline.py --calibrar                     # recalibra o limiar do golden template
   python src/avaliar.py                                 # avaliação no test/ (~3 min)

   # Etapa 2
   python src/particao.py                                # -> dados/particao.csv

   # Etapa 3
   python src/segmentacao.py                             # ~10 min na 1ª vez -> resultados/etapa3/

   # Etapa 4
   python src/impressao.py                               # figura de diagnóstico do "500"
   python src/atributos.py                               # ~1 min -> dados/atributos.csv
   ```

## 10. Estrutura do repositório

```
pdi-mvtec-capsule/
├── README.md
├── requirements.txt
├── docs/
│   └── projetos_pdi_quatro_propostas.pdf   enunciado
├── src/
│   ├── pipeline.py         Etapa 1: funções do pipeline e passo a passo de uma imagem
│   ├── avaliar.py          Etapa 1: avaliação nas 132 imagens de test/
│   ├── particao.py         Etapa 2: partição própria (banco / calibração / classificação)
│   ├── segmentacao.py      Etapa 3: máscara final (golden template + morfologia)
│   ├── impressao.py        Etapa 4: segmentação do "500" pelo histograma
│   └── atributos.py        Etapa 4: ficha de atributos de cada cápsula
├── dados/
│   ├── particao.csv        partição gerada (semente fixa)
│   ├── atributos.csv       fichas de atributos (Etapa 4)
│   └── gerado/             cache e máscaras da Etapa 3 (não versionado)
├── resultados/
│   ├── etapa1/             tabelas (avaliacao.txt) e exemplos de bordas e golden template
│   ├── etapa3/             tabela da morfologia e exemplos passo a passo
│   └── etapa4/             impressão "500" e distribuição dos atributos por tipo
└── capsule/                dataset MVTec AD (não versionado)
```
