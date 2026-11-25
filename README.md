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
| **7** | **Versão final.** Cascata de especialistas, um método por defeito (`src/especialistas.py`, `src/cascata.py`, `src/comparar_candidatos.py`, `src/classificacao_cascata.py`) | ✅ concluída; furinho refeito com LoG + Otsu no círculo + circularidade; todos os experimentos mínimos do enunciado refeitos nela |

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
| 4 | *poke* (furinho) | mancha **compacta e arredondada** | **Laplaciano da Gaussiana** (LoG) em 3 escalas, acima do envelope: os picos são os *blobs*; **Otsu num círculo** em volta de cada pico segmenta a mancha; **circularidade** 4πA/P² da região | resposta do LoG × circularidade (região ≥ 150 px) |
| 5 | *scratch* (arranhão) | feixe de **riscos finos** e claros, 3–10 níveis de contraste | *top-hat* de 9 px (sem o bilateral, que apaga os riscos), acima do envelope, limiar 3 | nº de pixels de fragmentos na janela 101×101 mais cheia |

**Envelope das cápsulas boas** (especialistas 4 e 5). A cápsula boa também tem estruturas finas
claras e escuras (linha de reflexo no meio, marca do molde, linhas internas), sempre no mesmo
lugar. O envelope é o **máximo** do *top-hat* (e do LoG, no especialista 4) nas 30 boas do banco,
dilatado 9 px; só o que passa dele conta. O que muda de lugar fica fora da **zona válida**: faixa de 13 px do contorno, o "500"
dilatado (é do especialista 2), a divisa preto/laranja e o **logo**, segmentado como uma cadeia
horizontal comprida de componentes claros (fechamento 45×15, largura ≥ 100 px), o que não apaga
um furinho isolado.

**Furinho: LoG + Otsu no círculo + circularidade (sugestão do professor).** A primeira versão do
especialista 4 media contraste local (*top-hat*/*black-hat* de 15 px, área da maior mancha). Ela
achava furinhos, mas também acusava arranhões, impressões falhas e rachaduras, e tudo o que achava
os outros especialistas também achavam: na ablação, tirá-la não perdia nenhuma cápsula. O que
distingue o furinho dos outros defeitos é a **forma** (mancha compacta, não risco fino), e a nova
versão mede isso em três passos (`resultados/etapa7/furinho_log_otsu.png`):

1. **Laplaciano da Gaussiana, σ = 6, 9 e 13 px**, normalizado em escala (σ²·∇²G). O LoG é um
   detector de *blobs*: responde no centro de manchas arredondadas de raio ≈ σ√2 (aqui, 17 a
   37 px de diâmetro) e pouco a riscos finos, porque a resposta a um risco de largura *w* cai
   com *w*/σ. A Gaussiana é calculada só com os pixels da zona de busca (média ponderada pelos
   pixels válidos, separada de cada lado da divisa), para que os degraus que não são defeito
   (fundo, tampa preta, "500", logo) não vazem para dentro dela. Como no *top-hat*, só conta o
   que passa do **envelope** das boas.
2. **Otsu no círculo.** Em volta de cada um dos 5 maiores picos, um círculo de raio 3σ. O Otsu
   dos tons de cinza **só dentro do círculo** separa a mancha da vizinhança dela (na cápsula
   inteira, o Otsu separaria preto × laranja). O LoG diz onde está o *blob* e se ele é claro ou
   escuro; o Otsu dá a região (fechamento 5 + abertura 3, componente que toca o centro).
3. **Circularidade** = 4πA/P² da região (1 = círculo; um risco que atravessa a janela dá ≈ 0,2).
   O valor da etapa é **resposta do LoG × circularidade**, só para regiões com **área ≥ 150 px**
   (os pontos de poeira das cápsulas boas são *blobs* perfeitos, mas têm 80–110 px).

A zona de busca do especialista 4 é a zona válida **menos o retângulo onde o "500" casou** com o
modelo do banco (o casamento do especialista 2): a máscara do "500" às vezes perde um pedaço do
"5", que viraria um *blob* claro. Dentro do retângulo da impressão quem responde é o
especialista 2.

**Correção da coluna da divisa.** Ao implementar o LoG apareceu um erro antigo: em 17 das 351
imagens a divisa preto/laranja era estimada na ponta esquerda da tampa (x ≈ 85 em vez de ≈ 515),
porque o brilho da ponta deixa as primeiras colunas claras, e a faixa da divisa não saía da zona
válida. A regra passou a ser "coluna seguinte à última majoritariamente preta" (idêntica nas
outras 334 imagens). Sozinha, a correção não muda nenhuma decisão da cascata em ALFA = 2%.

A pose da cápsula é estável na base (centro ±3 px, ângulo ±0,3°, tampa preta sempre à esquerda),
então o alinhamento com as referências é só uma translação pelo centroide; o ECC não é mais usado.

**Como cada método foi escolhido** (`src/comparar_candidatos.py`): para cada defeito, 3 a 5
candidatos foram comparados pelo AUC do defeito-alvo contra as boas e pela fração achada com ~2%
de alarme falso (tabela na seção 6). No furinho, o critério inclui também **quantas cápsulas dos
outros tipos** o candidato acusa.

**Limiar de cada etapa** (`src/cascata.py`), definido só com imagens de treino:
1. `a` = percentil (1 − ALFA) do valor nas boas de treino: a etapa aceita alarme falso em no
   máximo ALFA das boas;
2. limiar = ponto médio entre `a` e o menor valor, acima de `a`, entre os defeitos-alvo de treino.
   Onde há folga entre boas e defeitos (amassado, impressão), o limiar vai para o meio da folga.

**Protocolo:** referências = 30 boas do `banco`; limiares em validação cruzada com os 5 folds da
partição (treino = 20 boas de `calibracao` + 4 folds; decisão no 5º). ALFA é o ponto de operação:
2% (padrão) ou 5% (prioriza defeitos).

**Tipo apontado:** a primeira etapa que acusa. A ordem vai do especialista mais específico
(amassado, impressão) para os mais gerais: o furinho acusa qualquer mancha compacta (também as
lascas da rachadura) e o arranhão mede contraste local fino e reage a quase todos os defeitos.

**Refinamento do tipo pela diferença do "500".** A etapa 2 acusa qualquer mudança na impressão,
inclusive um furinho em cima dela (9 dos 21 furinhos saíam como "impressão falha"). O sinal da
diferença para o modelo casado diz o que mudou: na impressão falha **falta tinta**, e a cápsula
fica mais escura que o modelo; um furo sobre o "500" expõe material **mais claro**. Quando a
etapa 2 é a primeira a acusar e a fração clara da diferença (pixels com |diferença| > 30 em que
a cápsula é mais clara que o modelo) passa de um corte, o tipo apontado é furinho. O corte é
aprendido no treino de cada fold (o que melhor separa as impressões falhas dos furinhos que a
etapa 2 acusa; ≈ 0,17). Isso só muda o **nome** do defeito, não a decisão normal × defeituosa.

**Atributos e classificadores sobre os especialistas** (`src/classificacao_cascata.py`). Para
refazer na versão final as comparações do enunciado, a saída dos especialistas vira uma ficha
de atributos e vai para dois classificadores, com o protocolo da Etapa 5 (5 folds, padronização
e hiperparâmetros só no treino):

| Grupo | Atributos | Calculados sobre |
|---|---|---|
| Valores | os 5 valores dos especialistas | o que a cascata usa |
| Geometria | nº de manchas, área total, largura e altura da caixa; área, perímetro, compacidade, excentricidade, extensão, solidez, eixo maior e **orientação** da maior mancha | região suspeita = união das máscaras dos 5 especialistas (sem limiar de decisão) |
| Textura | média, desvio e contraste com um anel em volta; **cor** (média e desvio de a\* e b\*); histograma (suavidade, assimetria, uniformidade, entropia); LBP; **GLCM** (contraste, homogeneidade, energia, correlação); **HOG** (8 orientações do gradiente) | pixels da região suspeita |
| Posição | fração na tampa preta e sobre o "500", distância ao contorno, posição ao longo da cápsula | região suspeita |
| Diferença | área, fração clara, pixels claros fora dos traços e escuros sobre os traços | diferença do "500" para o modelo casado |
| Global (extensão opcional) | os mesmos descritores de textura | cápsula inteira |

Cor, orientação, GLCM e HOG estão no pipeline sugerido do enunciado e não tinham sido usados
(a GLCM tinha ficado de fora nas Etapas 4–5). Três tarefas: **binária** (normal × defeituosa),
**tipo** (qual dos 5 defeitos, nas 109 defeituosas) e **sistema** (a cascata decide e o
classificador dá o tipo), sempre comparadas com a regra da cascata.

**Efeito da morfologia e segmentações alternativas** (`src/comparar_candidatos.py`). Cada
especialista é rodado sem e com as suas operações morfológicas, e a máscara é medida pixel a
pixel (`tabela_morfologia.txt`). As duas segmentações do pipeline sugerido que faltavam entram
como candidatos: **k-means** (tampa da rachadura) e **crescimento de região** (furinho).

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
| | forma da tampa preta, k-means (2 grupos) em a\*, b\* | 0,754 | 0,52 |
| | **forma da tampa preta, Otsu no a\*** | **0,773** | **0,52** |
| | contraste local 15 px (método anterior do furinho) | 0,935 | 0,57 |
| | *blobs* do LoG + circularidade (especialista do furinho) | 0,844 | 0,65 |
| *poke* | método único | 0,562 | 0,05 |
| | top-hat/black-hat 15 px, sem envelope | 0,752 | 0,00 |
| | top-hat/black-hat 25 px, acima do envelope | 0,904 | 0,52 |
| | top-hat/black-hat 15 px, acima do envelope (anterior) | 0,961 | 0,62 |
| | anterior, só manchas com circularidade ≥ 0,4 | 0,922 | 0,33 |
| | LoG em 3 escalas, acima do envelope (maior pico) | 0,880 | 0,48 |
| | LoG + crescimento de região + circularidade | 0,845 | 0,38 |
| | **LoG + Otsu no círculo + circularidade** | **0,817** | **0,57** |
| *scratch* | método único | 0,492 | 0,00 |
| | pixels de borda do Canny (Etapa 1) | 0,539 | 0,04 |
| | maior risco (top-hat 9 px), acima do envelope | 0,961 | 0,61 |
| | concentração de fragmentos, sem envelope | 0,934 | 0,30 |
| | **concentração de fragmentos, acima do envelope** | **0,983** | **0,87** |

Na rachadura, o contraste local e os *blobs* do LoG têm AUC maior, mas os *blobs* já são a
etapa 4; a forma da tampa foi mantida como etapa 3 porque é ela que **dá nome** à rachadura
(acusa 12 das 23 antes que cheguem ao furinho).

No furinho, o AUC contra as boas **não** foi o critério: o método anterior ganha nele (0,961)
justamente porque reage a qualquer contraste local. O critério foi acusar o furinho **sem acusar
os outros defeitos** (cápsulas acima do percentil 98 das boas de treino, 5 folds):

| Candidato do furinho | boas (192) | crack (23) | imprint (22) | **poke (21)** | scratch (23) | squeeze (20) |
|---|---:|---:|---:|---:|---:|---:|
| top-hat/black-hat 15 px, acima do envelope (anterior) | 4 | 13 | 10 | **13** | 12 | 6 |
| anterior, só manchas com circularidade ≥ 0,4 | 2 | 10 | 3 | **7** | 3 | 3 |
| LoG em 3 escalas (maior pico) | 5 | 15 | 3 | **10** | 2 | 4 |
| LoG + crescimento de região + circularidade | 5 | 10 | 0 | **8** | 0 | 6 |
| **LoG + Otsu no círculo + circularidade** | 5 | 15 | 1 | **12** | 1 | 7 |

- O **LoG** é o que tira os arranhões (12 → 2) e as impressões falhas (10 → 3): riscos finos não
  formam *blob* nas escalas do furo.
- **Otsu no círculo + circularidade + área mínima** recuperam furinhos (10 → 12) e tiram mais
  impressões (3 → 1). Medir a circularidade direto na máscara do *top-hat*, sem o LoG, não
  funciona: a máscara sai picotada e os próprios furos ficam com circularidade baixa (13 → 7).
- Rachaduras e amassados continuam acusados (lascas e dobras também são manchas compactas), mas
  eles passam antes pelas etapas 1 e 3.
- O AUC menor contra as boas (0,817) vem dos furinhos que caem **sobre o "500"**, fora da zona de
  busca desta etapa (ficam com o especialista 2).
- As duas **segmentações do pipeline sugerido** que faltavam não ganham das escolhidas: o
  **k-means** na tampa dá quase a mesma máscara do Otsu no a\* (AUC 0,754 contra 0,773), e o
  **crescimento de região** a partir do pico do LoG acha menos furinhos que o Otsu no círculo
  (8 contra 12), porque o interior do furo não é homogêneo.

*Efeito da morfologia em cada especialista* (`tabela_morfologia.txt`; máscara sem o limiar de
decisão, pixel a pixel; nas boas todo pixel marcado é falso):

| Especialista | Operação | FP px/boa | Boas com pixel marcado | hit no alvo | Precisão | Dice | AUC | Achadas |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| amassado | XOR das silhuetas | 1.985 | 100% | 1,00 | 0,36 | 0,199 | 1,000 | 1,00 |
| | **+ abertura 7×7** | **19** | **27%** | 1,00 | 0,43 | 0,213 | 1,000 | 1,00 |
| impressão falha | \|teste − modelo\| > 30 | 1 | 8% | 0,73 | 0,97 | 0,093 | 1,000 | 1,00 |
| | + abertura 3×3 | 1 | 5% | 0,73 | 0,97 | 0,091 | 1,000 | 1,00 |
| rachadura | XOR das tampas | 3.261 | 100% | 0,78 | 0,23 | 0,137 | 0,729 | 0,17 |
| | **+ abertura 9×9** | **3** | **4%** | 0,57 | 0,65 | 0,181 | 0,773 | 0,52 |
| furinho | Otsu no círculo | 641 | 100% | 0,62 | 0,53 | 0,204 | 0,831 | 0,57 |
| | + fechamento 5 e abertura 3 | 617 | 100% | 0,62 | 0,54 | 0,205 | 0,817 | 0,57 |
| arranhão | fragmentos do *top-hat* | 60 | 95% | 1,00 | 0,80 | 0,122 | 0,985 | 0,87 |
| | + remoção < 4 px e fechamento 15 | 62 | 89% | 1,00 | 0,87 | 0,205 | 0,983 | 0,87 |

- A **abertura** é decisiva nos dois especialistas que comparam formas por XOR: entre duas
  cápsulas boas sobra uma casca fina ao longo do contorno, e a abertura a remove. Os pixels falsos
  por cápsula boa caem de 1.985 para 19 (amassado) e de 3.261 para 3 (rachadura), e as boas com
  algum pixel marcado, de 100% para 27% e 4%. Na rachadura isso é o que torna o valor útil
  (achadas 0,17 → 0,52), ao custo de perder as lascas mais finas (hit 0,78 → 0,57).
- No **arranhão**, remover os fragmentos menores que 4 px quase não muda os falsos positivos; o
  fechamento junta o feixe de riscos numa região e melhora a máscara (Dice 0,12 → 0,21).
- No **furinho** e na **impressão**, a morfologia não tem efeito mensurável: a região do furinho
  já sai limpa do Otsu no círculo, e a máscara da impressão quase não tem pixels soltos.
- A máscara do furinho marca 100% das boas porque é a região do melhor *blob*, exista defeito ou
  não; quem decide é o limiar da etapa. Com a decisão da cascata, sobram 19 px falsos por
  cápsula boa e 5% das boas com algum pixel marcado (`localizacao.txt`).

*Normal × defeituosa* (`tabela_cascata.txt`; 301 imagens, 5 folds):

| ALFA | Boas aprovadas | Defeituosas encontradas | Acur. bal. | Precisão | crack / imprint / poke / scratch / squeeze |
|---|---:|---:|---:|---:|---|
| 1% | 0,98 | 0,89 | 0,93 | 0,96 | 0,87 / 0,95 / 0,86 / 0,78 / 1,00 |
| **2% (padrão)** | **0,95** (183/192) | **0,91** (99/109) | **0,93** | **0,92** | 0,87 / 1,00 / 0,86 / 0,83 / 1,00 |
| 3% | 0,91 | 0,94 | 0,92 | 0,86 | 0,87 / 1,00 / 0,95 / 0,87 / 1,00 |
| **5% (prioriza defeitos)** | **0,87** (167/192) | **0,94** (103/109) | **0,91** | 0,80 | 0,91 / 1,00 / 0,95 / 0,87 / 1,00 |
| 8% | 0,83 | 0,95 | 0,89 | 0,76 | 0,91 / 1,00 / 0,95 / 0,91 / 1,00 |

ROC-AUC da cascata (maior valor/limiar entre as etapas): 0,97 (Etapa 5: 0,75).

Comparação com o método único (Etapa 5): acurácia balanceada **0,70 → 0,93**; encontrando 94% das
defeituosas, as boas aprovadas passam de **19% para 87%** (`ponto_de_operacao.png`).

*Furinho antes × depois* (`furinho_antes_depois.txt`; a mesma cascata, trocando só o valor da
etapa 4; aqui o tipo é o da regra simples, sem o refinamento pela diferença do "500"):

| | ALFA | Boas aprovadas | Defeituosas encontradas | Acur. bal. | Tipo apontado | Acerto do tipo: crack / imprint / poke / scratch / squeeze |
|---|---|---:|---:|---:|---:|---|
| anterior (top-hat 15 px) | 2% | 0,96 | 0,90 | 0,93 | 0,63 | 0,43 / 0,95 / 0,33 / 0,43 / 1,00 |
| **LoG + Otsu + circularidade** | 2% | 0,95 | 0,91 | 0,93 | **0,67** | 0,43 / 0,95 / 0,29 / **0,70** / 1,00 |
| anterior (top-hat 15 px) | 5% | 0,90 | 0,94 | 0,92 | 0,59 | 0,43 / 0,95 / 0,33 / 0,22 / 1,00 |
| **LoG + Otsu + circularidade** | 5% | 0,87 | 0,94 | 0,91 | **0,68** | 0,43 / 0,95 / 0,33 / **0,70** / 1,00 |

O que a etapa do furinho acusa (ALFA = 2%):

| | boa (192) | crack (23) | imprint (22) | poke (21) | scratch (23) | squeeze (20) | só ela acusa |
|---|---:|---:|---:|---:|---:|---:|---|
| anterior | 3 | 12 | 9 | 12 | 8 | 4 | 1 boa, nenhuma defeituosa |
| **depois** | 3 | 15 | **1** | 11 | **1** | 7 | 3 boas, 1 rachadura |

- **A etapa deixou de acusar o que não é mancha compacta**: arranhões 8 → 1 e impressões falhas
  9 → 1. O acerto do tipo "arranhão" sobe de 0,43 para **0,70** (7 arranhões eram chamados de
  furinho; agora 1), e o tipo apontado nas 5 classes, de 0,63 para **0,67**.
- **Ela deixou de ser redundante**: antes, tirá-la não perdia nenhuma defeituosa; agora perde 1
  rachadura que nenhuma outra etapa acha.
- **Custo:** os alarmes falsos dela não coincidem mais com os do arranhão (os dois mediam
  contraste local), então a cascata rejeita 2 boas a mais em ALFA = 2% (185 → 183) e, em
  ALFA = 5%, as boas aprovadas caem de 0,90 para 0,87 com o mesmo recall. A acurácia balanceada
  fica igual em 2% (0,93) e cai de 0,92 para 0,91 em 5%.
- **O que não mudou:** o acerto do próprio furinho (7 → 6 de 21). Dos 21 furinhos, 9 caem sobre
  o "500" ou encostados nele e são acusados antes pela etapa da impressão, 2 pela da rachadura,
  1 só pela do arranhão e 3 não são achados por nenhuma etapa (baixo contraste, junto ao
  contorno ou à divisa).

*Que etapa acusa cada tipo* (ALFA = 2%; `disparos_por_tipo.png`):

| Etapa | boa (192) | crack (23) | imprint (22) | poke (21) | scratch (23) | squeeze (20) |
|---|---:|---:|---:|---:|---:|---:|
| 1. amassado | 0 | 3 | 0 | 0 | 0 | **20** |
| 2. impressão falha | 1 | 1 | **21** | 9 | 2 | 7 |
| 3. rachadura | 0 | **12** | 0 | 4 | 0 | 18 |
| 4. furinho | 3 | 15 | 1 | **11** | 1 | 7 |
| 5. arranhão | 5 | 12 | 15 | 12 | **19** | 7 |
| alguma etapa | 9 | 20 | 22 | 18 | 19 | 20 |

- *Tipo apontado* (primeira etapa que acusa, refinada pela diferença do "500"): acurácia
  balanceada **0,71** nas 5 classes, contando defeituosa não achada como erro, e F1 macro 0,70
  (Etapa 5: 0,58, só entre defeituosas). Por tipo: *squeeze* 1,00, *faulty_imprint* 0,91,
  *scratch* 0,70, *poke* 0,52, *crack* 0,43. Entre as defeituosas achadas, o tipo está certo em
  78%. Sem o refinamento (só a primeira etapa): 0,67, com *poke* 0,29 e *faulty_imprint* 0,95;
  5 furinhos sobre o "500" deixam de ser chamados de impressão falha, ao custo de 1 impressão
  falha chamada de furinho.
- *Ablação* (`ablacao.txt`): tirar o especialista do arranhão derruba as defeituosas achadas de
  99 para 80 (16 arranhões, 1 impressão, 1 furinho e 1 rachadura); o da impressão, para 87
  (6 impressões, 5 furinhos e 1 rachadura); o do amassado, para 97; o da rachadura e o do
  furinho, para 98 (1 rachadura cada um). O arranhão ficou mais necessário (antes, parte dos
  arranhões também era acusada pelo furinho) e a rachadura, menos: as lascas que ela acha agora
  também são achadas como *blob* pelo furinho, mas é a etapa 3 que lhes dá o nome certo.
- *Localização* (`localizacao.txt`; união das máscaras das etapas que acusaram × ground truth):
  hit **0,87** e Dice **0,29** (Etapa 3: 0,64 e 0,155); nas boas, 19 px falsos por imagem e 5% com
  algum pixel marcado (Etapa 3: 1.913 px e 86%).
- *Curva ROC* (`roc_binaria.png`), *matriz de confusão do tipo* (`matriz_confusao_tipo.png`).
- *Erros* (`erros.csv`, `erro_*.png`): 9 boas rejeitadas (5 pelo arranhão, entre elas o logo
  cortado junto à borda; 3 pelo furinho, por um reflexo na ponta esquerda da tampa preta, com
  valor 3% a 9% acima do limiar; 1 pela impressão) e 10 defeituosas aprovadas (3 rachaduras,
  4 arranhões e 3 furinhos, de baixo contraste ou junto à divisa/contorno).

**Etapa 7 — atributos e classificadores sobre os especialistas** (`src/classificacao_cascata.py`)

*Normal × defeituosa* (`classificadores_binaria.txt`; 301 imagens, 5 folds, decisão padrão):

| Atributos | Decisão | Acur. bal. | Precisão | Recall | Especificidade | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| valores (5) | **cascata (um limiar por etapa)** | 0,931 | 0,917 | 0,908 | 0,953 | 0,912 | 0,969 |
| valores (5) | SVM | 0,925 | 0,941 | 0,881 | 0,969 | 0,910 | 0,979 |
| valores (5) | Random Forest | **0,940** | 0,918 | 0,927 | 0,953 | **0,922** | **0,982** |
| geometria (12) | SVM | 0,868 | 0,827 | 0,835 | 0,901 | 0,831 | 0,935 |
| geometria (12) | Random Forest | 0,875 | 0,823 | 0,853 | 0,896 | 0,838 | 0,949 |
| textura (25) | SVM | 0,881 | 0,867 | 0,835 | 0,927 | 0,850 | 0,952 |
| textura (25) | Random Forest | 0,871 | 0,835 | 0,835 | 0,906 | 0,835 | 0,938 |
| posição (4) | SVM | 0,727 | 0,660 | 0,642 | 0,812 | 0,651 | 0,782 |
| posição (4) | Random Forest | 0,735 | 0,670 | 0,651 | 0,818 | 0,660 | 0,808 |
| diferença (4) | SVM | 0,629 | 0,861 | 0,284 | 0,974 | 0,428 | 0,620 |
| diferença (4) | Random Forest | 0,632 | 0,886 | 0,284 | 0,979 | 0,431 | 0,616 |
| completo (50) | SVM | 0,920 | 0,884 | 0,908 | 0,932 | 0,896 | 0,978 |
| completo (50) | Random Forest | 0,923 | 0,892 | 0,908 | 0,938 | 0,900 | 0,981 |

*Qual defeito* (`classificadores_tipo.txt`; 109 defeituosas, 5 classes; acerto por tipo). Nas
regras da cascata, defeituosa que nenhuma etapa acusa conta como erro; os classificadores
recebem as 109 e dão um tipo a todas. A comparação nas mesmas condições é a do sistema completo,
logo abaixo.

| Atributos | Decisão | Acur. bal. | F1 macro | crack | imprint | poke | scratch | squeeze |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| valores | cascata: 1ª etapa que acusa | 0,674 | 0,678 | 0,43 | 0,95 | 0,29 | 0,70 | 1,00 |
| valores + diferença | **cascata: 1ª etapa + diferença do "500"** | 0,713 | 0,729 | 0,43 | 0,91 | 0,52 | 0,70 | 1,00 |
| valores (5) | SVM | 0,643 | 0,618 | 0,48 | 0,77 | 0,10 | 0,87 | 1,00 |
| valores (5) | Random Forest | 0,654 | 0,632 | 0,43 | 0,73 | 0,24 | 0,87 | 1,00 |
| geometria (12) | SVM | 0,326 | 0,327 | 0,30 | 0,23 | 0,38 | 0,22 | 0,50 |
| geometria (12) | Random Forest | 0,428 | 0,421 | 0,35 | 0,41 | 0,57 | 0,26 | 0,55 |
| textura (25) | SVM | 0,539 | 0,535 | 0,43 | 0,59 | 0,52 | 0,35 | 0,80 |
| textura (25) | Random Forest | 0,512 | 0,502 | 0,22 | 0,59 | 0,57 | 0,48 | 0,70 |
| posição (4) | SVM | 0,485 | 0,462 | 0,52 | 0,73 | 0,14 | 0,43 | 0,60 |
| posição (4) | Random Forest | 0,550 | 0,542 | 0,52 | 0,82 | 0,33 | 0,48 | 0,60 |
| diferença (4) | SVM | 0,348 | 0,268 | 0,00 | 0,59 | 0,00 | 1,00 | 0,15 |
| diferença (4) | Random Forest | 0,378 | 0,317 | 0,00 | 0,55 | 0,10 | 1,00 | 0,25 |
| completo (50) | SVM | 0,584 | 0,585 | 0,35 | 0,64 | 0,57 | 0,57 | 0,80 |
| completo (50) | Random Forest | **0,737** | **0,732** | 0,57 | 0,86 | 0,52 | 0,78 | 0,95 |

*Sistema completo* (`sistema_final.txt`): a cascata decide normal × defeituosa (ALFA = 2%) e o
tipo das acusadas vem de uma das regras ou do classificador; defeituosa não achada conta como
erro.

| Quem dá o tipo | Tipo apontado (acur. bal., 5 classes) | F1 macro | Tipo certo entre as achadas | crack / imprint / poke / scratch / squeeze |
|---|---:|---:|---:|---|
| regra simples: 1ª etapa que acusa | 0,674 | 0,653 | 74% | 0,43 / 0,95 / 0,29 / 0,70 / 1,00 |
| **regra adotada: 1ª etapa + diferença do "500"** | **0,713** | **0,702** | **78%** | 0,43 / 0,91 / 0,52 / 0,70 / 1,00 |
| Random Forest, ficha completa | 0,701 | 0,698 | 77% | 0,52 / 0,86 / 0,43 / 0,74 / 0,95 |

*O que cada atributo do pipeline sugerido acrescenta* (`ablacao_atributos.txt`; acurácia
balanceada):

| Atributos | nº | Binária SVM | Binária RF | Tipo SVM | Tipo RF |
|---|---:|---:|---:|---:|---:|
| completo | 50 | 0,920 | 0,923 | 0,584 | 0,737 |
| completo sem cor | 46 | 0,919 | 0,923 | 0,550 | 0,711 |
| completo sem orientação | 49 | 0,913 | 0,923 | 0,575 | 0,728 |
| completo sem GLCM | 46 | 0,928 | 0,929 | 0,576 | 0,772 |
| completo sem HOG | 42 | 0,906 | 0,927 | 0,579 | 0,701 |
| completo sem os 4 | 33 | 0,914 | 0,935 | 0,518 | 0,675 |
| completo sem diferença | 46 | 0,920 | 0,916 | 0,585 | 0,719 |
| valores + textura da região | 30 | 0,908 | 0,911 | 0,584 | 0,730 |
| valores + textura global | 29 | 0,904 | 0,955 | 0,686 | 0,757 |
| textura global, sozinha | 24 | 0,821 | 0,789 | 0,494 | 0,467 |

- **Classificadores × cascata, na detecção:** empate. O Random Forest sobre os 5 valores acha 2
  defeituosas a mais com as mesmas boas aprovadas (acur. bal. 0,940 contra 0,931) e ordena um
  pouco melhor (ROC-AUC 0,98 contra 0,97); o SVM fica em 0,925. Juntar os atributos da região
  não melhora a decisão (0,92). A cascata continua sendo a decisão adotada: é equivalente e cada
  limiar tem leitura direta.
- **SVM × Random Forest:** iguais na detecção; no tipo, o Random Forest é melhor com a ficha
  completa (0,74 contra 0,58), porque lida melhor com 50 atributos e ~20 exemplos por classe.
- **Grupos de atributos:** os valores dos especialistas sozinhos já dão 0,93–0,94 na detecção;
  geometria ou textura da região, sozinhas, 0,87–0,88. Para o **tipo**, nenhum grupo sozinho
  passa de 0,66, e a combinação completa chega a 0,74.
- **Quem dá o tipo:** a regra refinada pela diferença do "500" (0,71) e o Random Forest (0,70)
  empatam, e os dois ganham da regra simples (0,67). Ficou a regra: é um corte só, aprendido no
  treino, com explicação física (falta tinta × material exposto).
- **Cor, orientação, GLCM e HOG:** tirar qualquer um muda o resultado em menos de 0,04, dentro
  da variação esperada com ~20 cápsulas por tipo; tirar os quatro baixa o tipo de 0,74 para 0,68
  no Random Forest. Juntos ajudam um pouco a dizer o tipo; nenhum é decisivo.
- **Região × global (extensão opcional):** a textura da cápsula inteira, sozinha, é fraca (0,79
  a 0,82 na detecção), mas somada aos valores dos especialistas é a melhor ficha testada
  (detecção 0,955 e tipo 0,757 com Random Forest). É uma entre 15 fichas comparadas no mesmo
  conjunto, então o ganho precisa ser confirmado em cápsulas novas.

## 7. Discussão e limitações

- **Um método por defeito funciona muito melhor que um método único (Etapa 7).** Cada defeito
  deixa uma marca diferente: silhueta (amassado), impressão, cor da tampa (rachadura), mancha
  compacta (furinho) e riscos finos (arranhão). Procurar cada marca com o método certo levou a
  decisão normal × defeituosa de 0,70 para 0,93 de acurácia balanceada. O arranhão, invisível para
  o golden template (AUC 0,49), passa a ser separado das boas com AUC 0,98.
- **O envelope das boas é o que viabiliza o contraste local.** Sem descontar as estruturas normais,
  o mesmo top-hat acha 0% dos furinhos com 2% de alarme falso; com o envelope, 62%.
- **Medir a forma, e não só o contraste, tira a redundância do furinho.** O especialista antigo
  (contraste local) acusava quase todos os defeitos e não achava nada que os outros não achassem.
  Com LoG + Otsu no círculo + circularidade, ele acusa manchas compactas e quase não acusa riscos
  (arranhões 8 → 1, impressões 9 → 1); o tipo apontado sobe de 0,63 para 0,67 e o do arranhão, de
  0,43 para 0,70. O ganho é em **dizer o tipo**, não em detectar: a decisão boa × defeituosa fica
  igual em ALFA = 2% e perde 3 pontos de especificidade em 5%, porque um detector independente
  traz alarmes falsos próprios.
- **O sinal da diferença para a referência nomeia o furinho sobre o "500".** A etapa da
  impressão acusava 9 dos 21 furinhos. Na impressão falha falta tinta (mais escuro que o
  modelo); o furo expõe material mais claro. Com esse corte, o acerto do tipo "furinho" vai de
  0,29 para 0,52 e o tipo apontado, de 0,67 para 0,71.
- **Os especialistas ainda não são exclusivos.** O arranhão reage a quase todos os defeitos, e
  rachadura e furinho se confundem (lasca e furo são os dois manchas compactas): a rachadura
  continua sendo o tipo mais difícil de nomear (0,43).
- **Classificadores não superam a cascata.** Com os mesmos 5 valores, SVM e Random Forest
  empatam com os limiares da cascata na detecção (0,93–0,94), e o Random Forest com a ficha
  completa empata com a regra no tipo (0,70 contra 0,71). O que limita não é a regra de decisão,
  e sim o que os especialistas medem: as defeituosas com valor baixo em todos eles continuam
  difíceis para qualquer classificador (o melhor acha 101 das 109; a cascata, 99).
- **Morfologia (Etapa 7):** a abertura é o que viabiliza os especialistas que comparam formas
  (amassado e rachadura): sem ela, 100% das boas têm pixels marcados.

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
   tamanhos dos elementos estruturantes e os limiares internos (3 níveis, janela de 101 px). No
   furinho, as escalas do LoG (6, 9, 13 px), o raio do círculo (3σ), a área mínima (150 px), o
   produto resposta × circularidade e a exclusão do retângulo do "500" também foram escolhidos
   assim, depois de ver os resultados do método anterior e de algumas variantes (escalas 4 a
   18 px, área mínima 0 a 250 px, limiar de circularidade em vez do produto, tolerância do
   envelope 9 a 27 px: todas dão tipo apontado entre 0,65 e 0,68 em ALFA = 2%).
   Só os limiares de decisão das etapas são definidos em validação cruzada. Os números da Etapa 7
   são, portanto, otimistas; uma estimativa isenta exigiria cápsulas novas.
5. A Etapa 7 supõe a pose da base (tampa preta à esquerda, cápsula horizontal e centrada) e não
   resolve o logo cortado junto à borda nem o reflexo na ponta da tampa preta, as causas de
   alarme falso que restaram.
6. O especialista do furinho não procura dentro do retângulo do "500": furinhos sobre a impressão
   dependem do especialista 2 para serem achados; o tipo vem do refinamento pela diferença.
7. **Escolhas desta última rodada, também feitas olhando `classificacao`:** usar o sinal da
   diferença do "500" para o tipo (a ideia veio de inspecionar os 9 furinhos acusados pela etapa
   2; só o corte é aprendido no treino); a região suspeita como união das máscaras; a ficha e o
   classificador do "sistema" (completa + Random Forest) fixados antes de ver a tabela. Foram
   comparadas 15 fichas × 2 classificadores no mesmo conjunto: a melhor de cada tabela é
   otimista.

## 8. Próximos passos

1. ~~Partição própria~~ (Etapa 2, feita).
2. ~~Morfologia completa, quantificando o efeito sobre os falsos positivos~~ (Etapa 3, feita).
3. ~~Localização: tolerar a variação do "500" e do logo~~ (realinhamento local, feito). Ainda
   falta tratar contorno e reflexos, que geram as manchas falsas restantes.
4. ~~Atributos da região segmentada~~ (Etapa 4, feita; textura por estatísticas do histograma e
   LBP; a GLCM e o HOG entraram depois, na ficha da Etapa 7).
5. ~~Classificação com SVM e Random Forest, por grupo de atributos~~ (Etapa 5, feita).
6. ~~Relatório final~~ (`docs/relatorio.md`, versão inicial).
7. ~~Tratar cada defeito com um método específico, como foi feito com o "500"~~ (Etapa 7, feita:
   cascata de 5 especialistas).
8. ~~Tirar a redundância do especialista do furinho~~ (feito: LoG + Otsu no círculo +
   circularidade, por sugestão do professor).
9. ~~Refazer na versão final os experimentos mínimos do enunciado~~ (feito: morfologia por
   especialista, grupos de atributos, SVM × Random Forest × cascata, métricas por tipo) e
   ~~nomear o furinho sobre o "500"~~ (feito: sinal da diferença para o modelo).
10. **Possível continuação:** achar o furinho **sobre o "500"** sem depender da etapa da
   impressão, aplicando o LoG à diferença entre a impressão e o modelo casado; confirmar em
   cápsulas novas o ganho da textura global somada aos valores; casar o logo com modelos de cápsulas boas para tratar o logo cortado junto à borda;
   separar rachadura de furinho com atributos da região acusada (cor e posição na tampa).

## 9. Como executar

1. Baixe a categoria **capsule** do MVTec AD e extraia em `capsule/`, na raiz do repositório.
2. Instale as dependências:
   ```
   pip install -r requirements.txt     # opencv-python, numpy, matplotlib, scikit-learn
                                       # (scikit-learn: Etapa 5 e classificacao_cascata.py)
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
   python src/cascata.py --recalcular                    # ~2 min -> resultados/etapa7/
   python src/comparar_candidatos.py                     # ~4 min -> candidatos e morfologia
   python src/classificacao_cascata.py --recalcular      # ~10 min -> classificadores (depois da cascata)
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
│   ├── comparar_candidatos.py   Etapa 7: candidatos por defeito e efeito da morfologia
│   └── classificacao_cascata.py Etapa 7: atributos e classificadores sobre os especialistas
├── dados/
│   ├── particao.csv        partição gerada (semente fixa)
│   ├── atributos.csv       fichas de atributos (Etapa 4)
│   ├── especialistas.csv   valor de cada especialista por imagem (Etapa 7)
│   ├── atributos_cascata.csv   ficha da região suspeita de cada cápsula (Etapa 7)
│   ├── limiares_cascata.json   limiares finais da cascata (Etapa 7)
│   └── gerado/             cache e máscaras da Etapa 3 (não versionado)
├── resultados/
│   ├── etapa1/             tabelas (avaliacao.txt) e exemplos de bordas e golden template
│   ├── etapa3/             tabelas da morfologia e por tipo de defeito, exemplos passo a passo
│   ├── etapa4/             impressão "500" e distribuição dos atributos por tipo
│   ├── etapa5/             tabelas, matrizes de confusão, importância dos atributos e erros
│   └── etapa7/             candidatos, morfologia, cascata, ablação, furinho antes × depois,
│                           localização, classificadores, sistema final, exemplos e erros
└── capsule/                dataset MVTec AD (não versionado)
```
