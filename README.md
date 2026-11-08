# Projeto 4 — Detecção e classificação de defeitos industriais (MVTec AD · *capsule*)

Disciplina de Processamento Digital de Imagens — UFSCar · Prof. Dr. Ricardo J. Ferrari

> **Questão de projeto:** é possível detectar e classificar defeitos usando apenas processamento
> clássico e atributos interpretáveis?

## Sumário

1. [Contexto e objetivo](#1-contexto-e-objetivo)
2. [Dados e escopo](#2-dados-e-escopo)
3. [Andamento](#3-andamento)
4. [Pipeline da Etapa 1 — localização do defeito](#4-pipeline-da-etapa-1--localização-do-defeito)
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
└──────────── Etapa 1: localização ────────────┘ └─── Etapa 2: decisão ───┘
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
| **2** | Partição própria, morfologia completa, atributos e classificação | ⏳ próximos passos |

## 4. Pipeline da Etapa 1 — localização do defeito

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

## 5. Avaliação

Executada sobre as 132 imagens de `test/` (109 defeituosas + 23 boas), pixel a pixel contra o
ground truth:

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

Resultados completos em `resultados/avaliacao.txt`; figuras de exemplo em `resultados/*.png`.

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

**Limitações metodológicas:**
1. Os parâmetros das bordas e as tabelas de sensibilidade foram avaliados no próprio `test/`; falta
   uma **partição própria** de calibração e avaliação.
2. Uma única referência por imagem não tolera a variação normal entre cápsulas boas.

## 8. Próximos passos

1. **Partição própria:** separar cápsulas normais em referência/calibração e avaliar só no restante,
   sem escolher parâmetros no conjunto de avaliação.
2. **Localização:** tornar a referência tolerante à variação normal entre cápsulas boas.
3. **Morfologia completa:** fechamento, preenchimento de regiões e remoção de componentes pequenos, quantificando o efeito sobre os falsos positivos.
4. **Atributos** da região segmentada:
   - geometria: área, perímetro, extensão, orientação;
   - intensidade: médias, desvios, histogramas;
   - textura: LBP e GLCM.
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
3. Rode o pipeline ou a avaliação:
   ```
   python pipeline_etapa1.py --defeito crack --indice 0     # passo a passo de uma imagem
   python pipeline_etapa1.py --calibrar                     # recalibra o limiar do golden template
   python avaliar_etapa1.py                                 # avaliação no test/ (~3 min)
   ```

## 10. Estrutura do repositório

```
pdi-mvtec-capsule/
├── pipeline_etapa1.py      pipeline da Etapa 1 (uma imagem, com figuras do passo a passo)
├── avaliar_etapa1.py       avaliação nas 132 imagens de test/
├── requirements.txt
├── resultados/
│   ├── avaliacao.txt       tabelas de resultados
│   └── *.png               exemplos: bordas e golden template por tipo de defeito
└── capsule/                dataset (não versionado)
```
