# Projeto 4 — Detecção e classificação de defeitos industriais (MVTec AD · *capsule*)

Disciplina de Processamento Digital de Imagens — UFSCar · Prof. Dr. Ricardo J. Ferrari

> **Questão de projeto:** é possível detectar e classificar defeitos usando apenas processamento
> clássico e atributos interpretáveis?

## Sumário

1. [Contexto e objetivo](#1-contexto-e-objetivo)
2. [Dados e escopo](#2-dados-e-escopo)
3. [Andamento](#3-andamento)
4. [Pipeline (Etapas 1–5 e 7)](#4-pipeline-etapas-15-e-7)
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

O projeto tem duas abordagens. As Etapas 1–5 usam um **método único** para todos os defeitos
(golden template + ficha de atributos + classificador). A **Etapa 7**, por orientação do professor,
troca isso por **um método especialista para cada defeito**, em cascata: a cápsula passa por 5
etapas e só é aprovada se nenhuma acusar. É a abordagem com melhor resultado (seção 6).

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
| **3** | Máscara final: golden template com "500" e logo realinhados + morfologia completa (`src/segmentacao.py`) | ✅ concluída |
| **4** | Atributos: ficha da mancha mais suspeita, com "500" e logo segmentados pelo histograma (`src/impressao.py`, `src/atributos.py`) | ✅ concluída |
| **5** | Classificação: SVM × Random Forest, por grupo de atributos (`src/classificacao.py`) | ✅ concluída |
| **6** | Relatório final (`docs/relatorio.md`) | ✅ atualizado com a Etapa 7 |
| **7** | Cascata de especialistas, um método por defeito (`src/especialistas.py`, `src/cascata.py`, `src/comparar_candidatos.py`) | ✅ concluída |

## 4. Pipeline (Etapas 1–5 e 7)

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
Golden template da Etapa 1 com o **"500" e o logo realinhados localmente** e, depois, morfologia
completa, aplicada de forma cumulativa para medir o efeito de cada operação.

**Realinhamento local da impressão e do logo** (`src/impressao.py`, `diferenca_com_impressao`).
A cápsula gira em torno do próprio eixo, então o "500" e o logo aparecem alguns pixels deslocados
em relação à referência mesmo depois do alinhamento global, e a subtração acusava essa variação
normal. O "500" e o logo são segmentados pelo histograma do realce local (seção 4.7) no teste e
na referência; na região deles, o deslocamento local é estimado por **template matching** (busca
de ±20 px) e a diferença é recalculada com a referência deslocada. A variação de posição some;
uma falha de impressão continua aparecendo, porque muda a forma, não só a posição.

| Passo | Operação | Objetivo |
|---|---|---|
| C0 | \|teste − ref\| > T, com T = p99 da diferença nas 20 boas de calibração (= 27) | marcação inicial |
| C1 | abertura 5×5 | remover fragmentos finos (= Etapa 1) |
| C2 | ROI erodida 9 px | excluir o contorno da cápsula |
| C3 | fechamento 5×5 | unir pedaços do mesmo defeito |
| C4 | preenchimento de buracos (*flood fill*) | manchas sólidas |
| C5 | remoção de componentes conexos < A_MIN, com A_MIN = p95 da maior mancha nas boas de calibração (= 2.450 px) | remover manchas pequenas |

A máscara **C4** de cada imagem de `classificacao` é salva em `dados/gerado/mascaras/` para a etapa
de atributos (ver resultados na seção 6).

### 4.7 Etapa 4 — atributos (`src/impressao.py`, `src/atributos.py` → `dados/atributos.csv`)

**Impressão "500" pelo histograma (orientação do professor).** Dentro da máscara da cápsula, o
histograma separa metade preta (< 45), logo (45–75), metade laranja (75–112) e impressão (> 112).
Como a iluminação deixa a metade laranja com dois tons, o limiar é calculado no histograma do
**realce local** (cinza − mediana 45×45), só na metade clara e longe do contorno, por Otsu; depois,
abertura 5×5 e componentes conexos com forma de algarismo (`resultados/etapa4/impressao_500.png`).
O **logo** é segmentado do mesmo jeito, na metade preta, com um contraste mínimo (quando a
cápsula está girada, o logo pode não aparecer). As duas máscaras são usadas no realinhamento local
da Etapa 3 (seção 4.6). Na máscara do defeito, o "500" e o logo **não são apagados**, porque os
defeitos também caem sobre eles (sobre o "500": impressão falha ≈ 89% da área, furinho e arranhão
≈ 45%; sobre o logo: rachadura ≈ 14%, furinho ≈ 20%).

**Mancha principal = a mais suspeita.** Cada mancha da máscara C4 com metade ou mais da área sobre o
"500" ou o logo (dilatados 15 px) é tratada como variação normal; a mancha descrita pela ficha é a
maior entre as **outras** manchas. Motivo: depois da Etapa 3, ≈ 76% dos pixels falsos que sobram nas
cápsulas boas ficam sobre o logo, e a maior mancha costumava ser a do logo, escondendo o defeito
real numa mancha menor. As manchas sobre o "500" e o logo viram atributos próprios (área sobre cada
zona).

**Ficha (30 atributos por imagem, 301 imagens de `classificacao`)**, em 3 grupos para a comparação
do enunciado:

| Grupo | Atributos | Calculados sobre |
|---|---|---|
| Forma | nº de manchas, nº e área das manchas suspeitas, área, perímetro, compacidade (P²/4πA), excentricidade, extensão, solidez, momentos de Hu 1–3 | mancha principal |
| Textura | média e desvio do cinza, média e máximo da diferença ao modelo, suavidade, assimetria, uniformidade e entropia do histograma, entropia e uniformidade do LBP (8 vizinhos, raio 1) | pixels da mancha principal |
| Posição e impressão | fração na metade preta e distância ao contorno (mancha principal); área de mancha sobre o "500" e sobre o logo; área relativa do "500" (÷ mediana nas 20 boas de calibração), nº de componentes, nº de buracos e variação de área dos algarismos | manchas e impressão |

Cápsula sem mancha suspeita: atributos da mancha principal = 0 (e `n_suspeitas` = 0). Nenhum
atributo usa os rótulos.
Distribuições por tipo em `resultados/etapa4/atributos_por_tipo.png`: a área relativa e o nº de
buracos do "500" destacam a impressão falha; área, compacidade, diferença ao modelo e distância ao
contorno destacam o *squeeze*.

### 4.8 Etapa 5 — classificação (`src/classificacao.py` → `resultados/etapa5/`)

Duas tarefas em sequência, como numa linha de inspeção:
- **binária:** normal × defeituosa, nas 301 imagens de `classificacao`;
- **defeitos:** qual dos 5 defeitos, só nas 109 defeituosas.

Para cada tarefa, comparam-se os grupos de atributos (forma, textura, posição/impressão e todos)
e dois classificadores: **SVM** (núcleo RBF, atributos padronizados por z-score) e **Random
Forest** (300 árvores). Classes desbalanceadas são tratadas com `class_weight='balanced'`.

**Protocolo sem vazamento:** validação cruzada com a coluna `fold` da partição (5 rodadas: treina em
4 folds, testa no 5º), de modo que cada imagem é testada uma vez por um modelo que não a viu. A
padronização e a escolha de hiperparâmetros (busca em grade com validação interna de 3 partes;
SVM: C e γ; RF: profundidade e folhas mínimas) usam **só o treino de cada rodada**.

**Ponto de operação que prioriza os defeitos.** O objetivo é achar os defeitos: rejeitar uma
cápsula boa é um erro menor que aprovar uma defeituosa. Além da decisão padrão, a tarefa binária é
avaliada com um limiar sobre a pontuação "defeituosa" escolhido em cada rodada **só no treino**
(previsões fora da amostra da validação interna): o maior limiar que ainda encontra ≥ 90% das
defeituosas do treino.

### 4.9 Etapa 7 — um especialista por defeito, em cascata (`src/especialistas.py`, `src/cascata.py`)

**Orientação do professor:** em vez de procurar um método que ache todos os defeitos, usar um
método para cada defeito, como foi feito na Etapa 4 com a impressão "500". Cada imagem passa por
5 etapas; se nenhuma acusar, a cápsula é boa.

```
imagem → 1. amassado → 2. impressão falha → 3. rachadura → 4. furinho → 5. arranhão → BOA
              │                │                  │              │             │
              └────────────────┴──── acusou: DEFEITUOSA ─────────┴─────────────┘
```

Cada especialista procura só a marca que o seu defeito deixa na imagem e devolve **um número**
(quanto maior, mais suspeita) e a máscara da região acusada:

| # | Defeito | Marca na imagem | Método | Valor |
|---|---|---|---|---|
| 1 | *squeeze* (amassado) | a **silhueta** muda | máscara da cápsula (Otsu) × silhueta boa mais parecida do banco: XOR + abertura 7×7 | área da maior região que sobra |
| 2 | *faulty_imprint* | o **"500"** muda | cada algarismo é casado com o de cápsulas boas por correlação normalizada (*template matching*, busca de ±40/±60 px) | 1 − correlação do pior algarismo |
| 3 | *crack* (rachadura) | a tampa preta lasca e mostra o laranja, ou sai pó branco | segmentação preto × laranja pela **cor** (Otsu no canal a\* do Lab, dentro da cápsula); forma da tampa × tampa boa mais parecida, fora do "500": XOR + abertura 9×9 | área da maior região que sobra |
| 4 | *poke* (furinho) | mancha compacta com **contraste local** | *top-hat* e *black-hat* de 15 px, acima do envelope das boas, limiar 6, fechamento 5 + abertura 3 | área da maior mancha |
| 5 | *scratch* (arranhão) | feixe de **riscos finos** e claros, 3–10 níveis de contraste | *top-hat* de 9 px (sem o bilateral, que apaga os riscos), acima do envelope, limiar 3 | nº de pixels de fragmentos na janela 101×101 mais cheia |

**Envelope das cápsulas boas** (especialistas 4 e 5). A cápsula boa também tem estruturas finas
claras e escuras (linha de reflexo no meio, marca do molde, linhas internas), sempre no mesmo
lugar. O envelope é o **máximo** do *top-hat* nas 30 boas do banco, dilatado 9 px; só o que passa
dele conta. O que muda de lugar fica fora da **zona válida**: faixa de 13 px do contorno, o "500"
dilatado (é do especialista 2), a divisa preto/laranja e o **logo**, segmentado como uma cadeia
horizontal comprida de componentes claros (fechamento 45×15, largura ≥ 100 px), o que não apaga
um furinho isolado.

A pose da cápsula é estável na base (centro ±3 px, ângulo ±0,3°, tampa preta sempre à esquerda),
então o alinhamento com as referências é só uma translação pelo centroide; o ECC não é mais usado.

**Como cada método foi escolhido** (`src/comparar_candidatos.py`): para cada defeito, 3 a 5
candidatos foram comparados pelo AUC do defeito-alvo contra as boas e pela fração achada com ~2%
de alarme falso (tabela na seção 6).

**Limiar de cada etapa** (`src/cascata.py`), definido só com imagens de treino:
1. `a` = percentil (1 − ALFA) do valor nas boas de treino: a etapa aceita alarme falso em no
   máximo ALFA das boas;
2. limiar = ponto médio entre `a` e o menor valor, acima de `a`, entre os defeitos-alvo de treino.
   Onde há folga entre boas e defeitos (amassado, impressão), o limiar vai para o meio da folga.

**Protocolo:** referências = 30 boas do `banco`; limiares em validação cruzada com os 5 folds da
partição (treino = 20 boas de `calibracao` + 4 folds; decisão no 5º). ALFA é o ponto de operação:
2% (padrão) ou 5% (prioriza defeitos).

**Tipo apontado:** a primeira etapa que acusa. A ordem vai do especialista mais específico
(amassado, impressão) para os mais gerais (furinho, arranhão), que medem contraste local e também
reagem aos outros defeitos.

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

**Classificação (Etapa 5)**, por imagem, com as previsões das 5 rodadas da validação cruzada:

| Métrica | Significado |
|---|---|
| **Acurácia balanceada** | média do acerto de cada classe (não favorece a classe maior); acaso = 0,50 na binária e 0,20 nos 5 defeitos |
| **F1** | equilíbrio entre precisão e recall (binária: classe defeituosa; defeitos: média macro) |
| **ROC-AUC** | (binária) chance de uma defeituosa receber pontuação maior que uma normal |
| **Matriz de confusão** | quantas imagens de cada classe foram previstas como cada classe |

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

**Etapa 3 — efeito da morfologia** (`resultados/etapa3/`). Golden template com o "500" e o logo
realinhados localmente, parâmetros calibrados só nas 20 normais de `calibracao` (T = 27 = p99 da
diferença entre boas; A_MIN = 2.450 px = p95 da maior mancha que sobra em cada boa), avaliado nas
301 imagens de `classificacao`:

| Etapa (cumulativa) | hit | FP px/boa | Boas com alarme | Precisão | Dice |
|---|---:|---:|---:|---:|---:|
| C0 limiar | 0,68 | 2.416 | 99% | 0,31 | 0,155 |
| C1 + abertura 5 | 0,65 | 1.905 | 86% | 0,34 | 0,159 |
| C2 + ROI erodida 9 px | 0,64 | 1.832 | 86% | 0,34 | 0,155 |
| C3 + fechamento 5 | 0,64 | 1.856 | 86% | 0,34 | 0,156 |
| C4 + preenchimento | 0,64 | 1.913 | 86% | 0,34 | 0,155 |
| C5 + remoção de componentes < A_MIN | 0,17 | 568 | 10% | 0,45 | 0,058 |

Efeito do realinhamento local do "500" e do logo (máscara C4, mesmas imagens):

| | hit | FP px/boa | Dice | hit *crack* | hit *faulty_imprint* | hit *scratch* |
|---|---:|---:|---:|---:|---:|---:|
| sem realinhamento (T = 32) | 0,57 | 2.398 | 0,126 | 0,65 | 0,59 | 0,09 |
| **com realinhamento (T = 27)** | **0,64** | **1.913** | **0,155** | **0,83** | **0,82** | 0,00 |

- O realinhamento local reduz a variação normal nas boas (o limiar calibrado cai de 32 para 27) e
  aumenta a detecção, sobretudo de impressão falha e rachadura.
- A abertura é a única operação morfológica que reduz falsos positivos sem custo adicional
  relevante; ROI erodida, fechamento e preenchimento mudam pouco os números.
- A remoção por área elimina quase todos os alarmes nas boas, mas também quase todos os defeitos:
  as manchas falsas das cápsulas boas têm o **mesmo tamanho** dos defeitos. Por isso a máscara
  levada para a Etapa 4 é a **C4**; separar "mancha falsa" de "defeito" fica a cargo dos atributos e
  do classificador. (Escolha feita depois de ver a tabela e declarada aqui.)
- Nas boas, ≈ 76% dos pixels falsos que sobram em C4 estão sobre o logo, ≈ 13% junto ao contorno e
  só ≈ 3% sobre o "500".

**Etapa 5 — classificação** (`resultados/etapa5/`):

*Normal × defeituosa* (301 imagens), decisão padrão:

| Grupo | SVM: acur. bal. / F1 / AUC | Random Forest: acur. bal. / F1 / AUC |
|---|---|---|
| forma | 0,59 / 0,33 / 0,64 | 0,64 / 0,53 / 0,65 |
| textura | 0,58 / 0,45 / 0,65 | 0,59 / 0,45 / 0,63 |
| posição/impressão | 0,66 / 0,55 / 0,69 | 0,70 / 0,60 / 0,74 |
| **todos** | **0,70 / 0,60 / 0,75** | 0,70 / 0,60 / 0,75 |

Todos + RF: precisão 0,70, recall 0,52, especificidade 0,87 (167 de 192 boas aprovadas, 57 de 109
defeituosas encontradas).

*Normal × defeituosa*, **priorizando os defeitos** (limiar para recall ≥ 90% no treino):

| Grupo | SVM: recall / especificidade | Random Forest: recall / especificidade |
|---|---|---|
| forma | 0,91 / 0,13 | 0,94 / 0,06 |
| textura | 0,92 / 0,21 | 0,95 / 0,12 |
| posição/impressão | 0,92 / 0,18 | 0,94 / 0,18 |
| **todos** | **0,94 / 0,19** | 0,94 / 0,19 |

Todos + SVM: 102 de 109 defeituosas encontradas (*crack* 21/23, *faulty_imprint* 22/22, *poke*
19/21, *scratch* 21/23, *squeeze* 19/20), ao custo de aprovar só 37 das 192 boas (precisão 0,40).

*Qual defeito* (109 defeituosas, 5 classes):

| Grupo | SVM: acur. bal. / F1 macro | Random Forest: acur. bal. / F1 macro |
|---|---|---|
| forma | 0,39 / 0,39 | 0,38 / 0,38 |
| textura | 0,53 / 0,53 | 0,45 / 0,43 |
| posição/impressão | 0,33 / 0,34 | 0,50 / 0,49 |
| **todos** | 0,55 / 0,54 | **0,58 / 0,57** |

Acerto por tipo (todos + RF): *squeeze* 0,90, *faulty_imprint* 0,86, *crack* 0,43, *scratch* 0,43,
*poke* 0,29.

Evolução da classificação com as mudanças de localização e de ficha (melhor configuração; com
~20 exemplos por defeito, diferenças de até ±0,05 estão dentro da variação esperada):

| Versão | Normal × defeituosa (acur. bal.) | Prioriza defeitos (recall / espec.) | Qual defeito (acur. bal.) |
|---|---:|---:|---:|
| ficha da maior mancha, sem realinhamento | 0,70 | 0,94 / 0,29 | 0,53 |
| + realinhamento do "500"/logo | 0,71 | 0,93 / 0,23 | 0,48 |
| **+ mancha principal fora do "500"/logo (final)** | **0,70** | **0,94 / 0,19** | **0,58** |

**Etapa 7 — cascata de especialistas** (`resultados/etapa7/`)

*Qual método para cada defeito* (`tabela_candidatos.txt`; AUC do defeito-alvo × boas; "achadas" =
fração do defeito-alvo acima do percentil 98 das boas de treino, em 5 folds):

| Defeito | Método | AUC | Achadas |
|---|---|---:|---:|
| *squeeze* | método único (maior mancha do golden template, Etapas 3–4) | 0,992 | 0,95 |
| | descritores globais da silhueta | 0,993 | 0,95 |
| | **XOR com a silhueta boa mais parecida + abertura 7** | **1,000** | **1,00** |
| *faulty_imprint* | método único | 0,666 | 0,05 |
| | desvio da área do "500" (atributo da Etapa 4) | 0,817 | 0,36 |
| | correlação do "500" inteiro | 0,999 | 0,95 |
| | **correlação do pior algarismo** | **1,000** | **1,00** |
| *crack* | método único | 0,726 | 0,26 |
| | forma da tampa preta, Otsu no cinza | 0,636 | 0,13 |
| | **forma da tampa preta, Otsu no a\*** | **0,773** | **0,52** |
| | contraste local 15 px (especialista do furinho) | 0,935 | 0,57 |
| *poke* | método único | 0,562 | 0,05 |
| | top-hat/black-hat 15 px, sem envelope | 0,751 | 0,00 |
| | top-hat/black-hat 25 px, acima do envelope | 0,906 | 0,57 |
| | **top-hat/black-hat 15 px, acima do envelope** | **0,967** | **0,62** |
| *scratch* | método único | 0,492 | 0,00 |
| | pixels de borda do Canny (Etapa 1) | 0,526 | 0,04 |
| | maior risco (top-hat 9 px), acima do envelope | 0,961 | 0,61 |
| | concentração de fragmentos, sem envelope | 0,934 | 0,30 |
| | **concentração de fragmentos, acima do envelope** | **0,983** | **0,87** |

Na rachadura, o contraste local tem AUC maior, mas ele já é a etapa 4; a forma da tampa foi
escolhida porque acha as lascas, que quase não têm contraste local (sem a etapa 3, a cascata cai
de 19 para 14 rachaduras achadas).

*Normal × defeituosa* (`tabela_cascata.txt`; 301 imagens, 5 folds):

| ALFA | Boas aprovadas | Defeituosas encontradas | Acur. bal. | Precisão | crack / imprint / poke / scratch / squeeze |
|---|---:|---:|---:|---:|---|
| 1% | 0,98 | 0,88 | 0,93 | 0,96 | 0,83 / 0,95 / 0,86 / 0,78 / 1,00 |
| **2% (padrão)** | **0,96** (185/192) | **0,90** (98/109) | **0,93** | **0,93** | 0,83 / 1,00 / 0,86 / 0,83 / 1,00 |
| 3% | 0,94 | 0,93 | 0,93 | 0,89 | 0,87 / 1,00 / 0,90 / 0,87 / 1,00 |
| **5% (prioriza defeitos)** | **0,90** | **0,94** (103/109) | **0,92** | 0,84 | 0,96 / 1,00 / 0,90 / 0,87 / 1,00 |
| 8% | 0,86 | 0,98 | 0,92 | 0,80 | 1,00 / 1,00 / 0,95 / 0,96 / 1,00 |

ROC-AUC da cascata (maior valor/limiar entre as etapas): 0,98 (Etapa 5: 0,75).

Comparação com o método único (Etapa 5): acurácia balanceada **0,70 → 0,93**; encontrando 94% das
defeituosas, as boas aprovadas passam de **19% para 90%** (`ponto_de_operacao.png`).

*Que etapa acusa cada tipo* (ALFA = 2%; `disparos_por_tipo.png`):

| Etapa | boa (192) | crack (23) | imprint (22) | poke (21) | scratch (23) | squeeze (20) |
|---|---:|---:|---:|---:|---:|---:|
| 1. amassado | 0 | 3 | 0 | 0 | 0 | **20** |
| 2. impressão falha | 1 | 1 | **21** | 9 | 2 | 7 |
| 3. rachadura | 0 | **12** | 0 | 4 | 0 | 18 |
| 4. furinho | 3 | 12 | 9 | **12** | 8 | 4 |
| 5. arranhão | 5 | 12 | 15 | 12 | **19** | 7 |
| alguma etapa | 7 | 19 | 22 | 18 | 19 | 20 |

- *Tipo apontado* (primeira etapa que acusa): acurácia balanceada **0,63** nas 5 classes, contando
  defeituosa não achada como erro (Etapa 5: 0,58, só entre defeituosas). Por tipo: *squeeze* 1,00,
  *faulty_imprint* 0,95, *crack* 0,43, *scratch* 0,43, *poke* 0,33.
- *Ablação* (`ablacao.txt`): tirar o especialista da impressão derruba a impressão falha de 22
  para 15 achadas (e os furinhos de 18 para 14); o do arranhão, de 19 para 9; o da rachadura, de
  19 para 14; o do amassado, de 20 para 18. Tirar o do furinho **não perde nenhuma cápsula** neste
  ponto de operação: os furinhos que ele acha também são achados pelos especialistas do arranhão
  e da impressão.
- *Localização* (`localizacao.txt`; união das máscaras das etapas que acusaram × ground truth):
  hit **0,86** e Dice **0,30** (Etapa 3: 0,64 e 0,155); nas boas, 20 px falsos por imagem e 4% com
  algum pixel marcado (Etapa 3: 1.913 px e 86%).
- *Erros* (`erros.csv`, `erro_*.png`): 7 boas rejeitadas (nas conferidas, o logo cortado junto à
  borda) e 11 defeituosas aprovadas (4 rachaduras, 4 arranhões e 3 furinhos, de baixo contraste
  ou junto à divisa/contorno).

## 7. Discussão e limitações

- **Um método por defeito funciona muito melhor que um método único (Etapa 7).** Cada defeito
  deixa uma marca diferente: silhueta (amassado), impressão, cor da tampa (rachadura), contraste
  local (furinho) e riscos finos (arranhão). Procurar cada marca com o método certo levou a
  decisão normal × defeituosa de 0,70 para 0,93 de acurácia balanceada. O arranhão, invisível para
  o golden template (AUC 0,49), passa a ser separado das boas com AUC 0,98.
- **O envelope das boas é o que viabiliza o contraste local.** Sem descontar as estruturas normais,
  o mesmo top-hat acha 0% dos furinhos com 2% de alarme falso; com o envelope, 62%.
- **Os especialistas não são exclusivos.** Furinho e arranhão medem contraste local e reagem a
  quase todos os defeitos; por isso a cascata acha bem **se há** defeito, mas aponta **qual** com
  acerto limitado (0,63). Rachadura, furinho e arranhão se confundem entre si, e o especialista
  do furinho acaba redundante na cascata (ablação).

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
- **Classificação (Etapa 5):** combinar os grupos é o que funciona melhor nas duas tarefas. Na
  decisão boa × defeituosa, o grupo posição/impressão sozinho já chega perto do total; para dizer
  **qual** defeito, a textura pesa mais (`importancia_atributos.png`). Descrever a mancha mais
  suspeita, fora do "500" e do logo, melhorou a identificação do tipo (0,53 → 0,58; rachadura
  0,22 → 0,43; impressão falha 0,55 → 0,86).
- **Achar defeitos custa aprovar poucas boas.** Para encontrar ≥ 90% das defeituosas, só ~19% das
  boas são aprovadas: mesmo depois do realinhamento, ≈ 75% das cápsulas boas ainda têm alguma
  mancha suspeita fora do "500" e do logo (contorno, reflexos, logo parcialmente visível), parecida
  com as manchas pequenas de arranhões e rachaduras. A melhora na localização (hit 0,57 → 0,64) não
  se converteu em melhor decisão boa × defeituosa (≈ 0,70 em todas as versões).
- **O gargalo continua sendo a localização (parte 1):** a decisão final depende de a variação
  normal das cápsulas boas não gerar manchas parecidas com defeitos. Exemplos de erro em
  `exemplos_erros.png` e `erros_binaria.csv`.
- **Morfologia (Etapa 3):** só a abertura reduz falsos positivos de forma útil. Filtrar por área não
  funciona, porque as manchas falsas da impressão têm o tamanho dos defeitos. A ROI erodida também
  corta parte de defeitos que ficam na borda da cápsula (*squeeze*).

**Limitações metodológicas:**
1. Na Etapa 1, os parâmetros das bordas e as tabelas de sensibilidade foram avaliados no próprio
   `test/`. A partir da Etapa 2 isso foi corrigido com a partição própria.
2. **Escolhas feitas depois de ver resultados no conjunto `classificacao`** (declaradas): usar C4
   em vez de C5; acrescentar o realinhamento do "500"/logo; e descrever a mancha principal fora do
   "500"/logo. As duas últimas foram motivadas por diagnósticos (onde estavam os pixels falsos), e
   cada uma foi rodada uma única vez; ainda assim, os números finais podem estar levemente
   otimistas.
3. Uma única referência por imagem não tolera toda a variação normal entre cápsulas boas (logo
   parcialmente visível, reflexos, contorno).
4. **Etapa 7: os especialistas foram desenhados olhando as imagens de `classificacao`** (todas as
   defeituosas estão nesse conjunto): a marca procurada em cada defeito, as zonas excluídas, os
   tamanhos dos elementos estruturantes e os limiares internos (6 e 3 níveis, janela de 101 px).
   Só os limiares de decisão das etapas são definidos em validação cruzada. Os números da Etapa 7
   são, portanto, otimistas; uma estimativa isenta exigiria cápsulas novas.
5. A Etapa 7 supõe a pose da base (tampa preta à esquerda, cápsula horizontal e centrada) e não
   resolve o logo cortado junto à borda, principal causa de alarme falso que restou.

## 8. Próximos passos

1. ~~Partição própria~~ (Etapa 2, feita).
2. ~~Morfologia completa, quantificando o efeito sobre os falsos positivos~~ (Etapa 3, feita).
3. ~~Localização: tolerar a variação do "500" e do logo~~ (realinhamento local, feito). Ainda
   falta tratar contorno e reflexos, que geram as manchas falsas restantes.
4. ~~Atributos da região segmentada~~ (Etapa 4, feita; textura por estatísticas do histograma e
   LBP, sem GLCM, que não foi vista em aula).
5. ~~Classificação com SVM e Random Forest, por grupo de atributos~~ (Etapa 5, feita).
6. ~~Relatório final~~ (`docs/relatorio.md`, versão inicial).
7. ~~Tratar cada defeito com um método específico, como foi feito com o "500"~~ (Etapa 7, feita:
   cascata de 5 especialistas).
8. **Possível continuação:** casar o logo com modelos de cápsulas boas (como o "500") para tratar
   o logo cortado junto à borda; separar melhor rachadura, furinho e arranhão com atributos da
   região acusada (forma e textura) e um classificador só para o tipo.

## 9. Como executar

1. Baixe a categoria **capsule** do MVTec AD e extraia em `capsule/`, na raiz do repositório.
2. Instale as dependências:
   ```
   pip install -r requirements.txt     # opencv-python, numpy, matplotlib, scikit-learn
                                       # (scikit-learn só é usado na Etapa 5)
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

   # Etapa 5
   python src/classificacao.py                           # ~6 min -> resultados/etapa5/

   # Etapa 7 (independente das Etapas 3-5; só precisa de dados/particao.csv)
   python src/cascata.py                                 # ~1 min -> resultados/etapa7/
   python src/comparar_candidatos.py                     # ~1 min -> tabela_candidatos.txt
   python src/especialistas.py --defeito crack --indice 10   # passo a passo de uma imagem
   ```

## 10. Estrutura do repositório

```
pdi-mvtec-capsule/
├── README.md
├── requirements.txt
├── docs/
│   ├── projetos_pdi_quatro_propostas.pdf   enunciado
│   └── relatorio.md        relatório final (Etapa 6)
├── src/
│   ├── pipeline.py         Etapa 1: funções do pipeline e passo a passo de uma imagem
│   ├── avaliar.py          Etapa 1: avaliação nas 132 imagens de test/
│   ├── particao.py         Etapa 2: partição própria (banco / calibração / classificação)
│   ├── segmentacao.py      Etapa 3: máscara final (golden template + morfologia)
│   ├── impressao.py        "500" e logo pelo histograma + realinhamento local (Etapas 3–4)
│   ├── atributos.py        Etapa 4: ficha de atributos de cada cápsula
│   ├── classificacao.py    Etapa 5: SVM e Random Forest com validação cruzada
│   ├── especialistas.py    Etapa 7: os 5 especialistas (um método por defeito)
│   ├── cascata.py          Etapa 7: limiares, decisão em cascata e avaliação
│   └── comparar_candidatos.py   Etapa 7: comparação dos métodos candidatos por defeito
├── dados/
│   ├── particao.csv        partição gerada (semente fixa)
│   ├── atributos.csv       fichas de atributos (Etapa 4)
│   ├── especialistas.csv   valor de cada especialista por imagem (Etapa 7)
│   ├── limiares_cascata.json   limiares finais da cascata (Etapa 7)
│   └── gerado/             cache e máscaras da Etapa 3 (não versionado)
├── resultados/
│   ├── etapa1/             tabelas (avaliacao.txt) e exemplos de bordas e golden template
│   ├── etapa3/             tabelas da morfologia e por tipo de defeito, exemplos passo a passo
│   ├── etapa4/             impressão "500" e distribuição dos atributos por tipo
│   ├── etapa5/             tabelas, matrizes de confusão, importância dos atributos e erros
│   └── etapa7/             candidatos, cascata, ablação, localização, exemplos e erros
└── capsule/                dataset MVTec AD (não versionado)
```
