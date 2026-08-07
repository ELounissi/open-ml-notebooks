[English](../README.md) · [العربية](README.ar.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Português](README.pt.md) · [हिन्दी](README.hi.md) · [简体中文](README.zh-CN.md)

# open-ml-notebooks

### Aprenda machine learning lendo o código

Um livro gratuito e aberto com **86 notebooks Jupyter** sobre aprendizado de
máquina, deep learning e aprendizado por reforço: um método por notebook,
explicado em linguagem simples, implementado do zero e depois feito direito, com
a biblioteca.

Todos os notebooks já vêm **executados**. Abra um no GitHub e os gráficos, os
números e as saídas estão ali. Não precisa instalar nada para começar a ler.

**Feito por [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)**

Os notebooks estão em inglês, e isso quase não atrapalha. O código, os nomes das
variáveis, as tabelas impressas e os rótulos dos gráficos estão no idioma que
toda biblioteca de ML e toda mensagem de erro falam de qualquer jeito, e gráficos
e números se leem igual em qualquer língua. Dá para acompanhar cada resultado sem
ler inglês com fluência.

![Uma reta não enxerga os dois picos do dia](../02-regression/01-linear-regression/figures/fig-04-where-it-loses.png)

*De [Regressão linear](../02-regression/01-linear-regression/): por que o modelo
mais simples falha com dados cíclicos, e o conserto de uma linha que o leva de
0,39 para 0,68.*

---

## Por que este livro

A maioria dos tutoriais mostra qual função chamar. Poucos mostram o que o método
está fazendo de fato, e quase nenhum diz **quando ele é a escolha errada**.

Cada notebook aqui responde a quatro perguntas:

1. **O que este método faz, afinal?** Primeiro em palavras, com uma figura.
2. **Qual é a matemática?** Escrita, só o que você precisa.
3. **Consigo construir sozinho?** Uma versão mínima em NumPy, para que nada seja mágica.
4. **Quando ele ganha e quando perde?** Medido, não afirmado.

A quarta pergunta é a que importa e a que costuma faltar.

## O fio que amarra tudo

Todos os notebooks usam **os mesmos cinco conjuntos de dados**. Isso é
proposital. Quando uma máquina de vetores de suporte tira 0,93 no capítulo três e
um Random Forest tira 0,94 no capítulo quatro, os números são diretamente
comparáveis, porque a pergunta era a mesma.

| Conjunto de dados | Tarefa | Tamanho | Por que este |
|---|---|---|---|
| California Housing | regressão | 20.640 × 8 | Vem junto com o scikit-learn: o capítulo um roda sem baixar nada |
| Breast Cancer Wisconsin | classificação binária | 569 × 30 | Pequeno o bastante para qualquer método treinar na hora |
| **UCI Dry Bean** | classificação em 7 classes | 13.611 × 16 | Publicado em 2020, quase não aparece em tutoriais e é agradavelmente desbalanceado |
| UCI Bike Sharing | regressão e séries temporais | 17.379 × 16 | Um conjunto só que serve aos capítulos tabulares e aos de sequências |
| Fashion-MNIST | classificação de imagens | 70.000 imagens | O formato do MNIST, várias vezes mais difícil, para as CNNs continuarem interessantes |

No fim, [**o placar**](CURRICULUM.pt.md) coloca cada método contra cada conjunto
de dados em uma tabela só e explica o padrão: por que gradient boosting costuma
levar os dados tabulares, por que os k vizinhos mais próximos desandam quando as
colunas se multiplicam, por que um modelo linear ganha de uma rede neural quando
há poucas linhas.

---

## Conteúdo

Sumário completo com os 86 notebooks: **[CURRICULUM.pt.md](CURRICULUM.pt.md)**

| Parte | O que cobre |
|---|---|
| **1: [Fundamentos](../01-foundations/)** | O fluxo de trabalho, treino/validação/teste, overfitting, validação cruzada, métricas, escalonamento, codificação, dados faltantes, ajuste |
| **2: [Regressão](../02-regression/)** | Linear, gradiente descendente, polinomial, Ridge, Lasso, Elastic Net, Huber, RANSAC, quantílica, GLMs |
| **3: [Classificação](../03-classification/)** | Regressão logística, k-NN, Naive Bayes, LDA/QDA, SVM e kernels, árvores de decisão, desbalanceamento, calibração |
| **4: [Ensembles](../04-ensembles/)** | Bagging, Random Forest, Extra Trees, AdaBoost, gradient boosting, XGBoost, LightGBM, CatBoost, stacking |
| **5: [Aprendizado não supervisionado](../05-unsupervised/)** | k-Means, escolher k, hierárquico, DBSCAN, HDBSCAN, misturas gaussianas, detecção de anomalias, Apriori |
| **6: [Redução de dimensionalidade](../06-dimensionality-reduction/)** | PCA, Kernel PCA, ICA, NMF, t-SNE, UMAP, seleção de variáveis |
| **7: [Redes neurais](../07-neural-networks/)** | Perceptron, MLP e retropropagação em NumPy, PyTorch, ativações, otimizadores, regularização |
| **8: [Visão computacional](../08-computer-vision/)** | Convolução, CNNs, LeNet/VGG/ResNet, aumento de dados, transfer learning, segmentação, detecção |
| **9: [Sequências e linguagem](../09-sequences-and-language/)** | TF-IDF, embeddings, RNN, LSTM, GRU, seq2seq, atenção, Transformers, séries temporais |
| **10: [Modelos generativos](../10-generative-models/)** | Autoencoders, VAEs, GANs, difusão |
| **11: [Aprendizado por reforço](../11-reinforcement-learning/)** | Bandidos, MDPs, Q-learning, SARSA, DQN, gradientes de política, ator-crítico, PPO |
| **12: [Juntando tudo](../12-putting-it-together/)** | O placar, SHAP e LIME, pipelines, deploy, erros comuns |

### Alguns que valem a pena abrir primeiro

- **[01-03: Overfitting e underfitting](../01-foundations/03-overfitting-and-underfitting/)**,
  o diagnóstico mais útil do aprendizado de máquina, desenhado em vez de
  definido, incluindo o ajuste de grau 20 que sai pior que o de grau 4 tanto no
  treino quanto no teste, porque a matriz de design tem número de condição
  `1.1e+21` e o solver desiste caladinho antes de o overfitting ter chance
- **[02-01: Regressão linear](../02-regression/01-linear-regression/)**, a
  equação normal deduzida, uma versão de 30 linhas em NumPy batendo com o
  scikit-learn até a 12ª casa decimal, diagnóstico de resíduos que expõe um viés
  que a estatística resumo escondia, e o método falhando em dados cíclicos e
  depois salvo pela codificação, não por um modelo maior
- **[02-04: Regressão Ridge](../02-regression/04-ridge-regression/)**, a
  penalidade L2 e a constatação honesta de que ela não trouxe **nenhuma
  acurácia** nestes dados, embora tenha deixado os coeficientes **32× mais
  estáveis**; junto com um experimento de escalonamento que saiu ao contrário e se
  explica sozinho
- **[02-05: Regressão Lasso](../02-regression/05-lasso-regression/)**, por que um
  único expoente transforma encolhimento em seleção, e a descoberta de que **o
  `LassoCV` manteve 17 de 30 colunas de puro ruído**, porque a validação cruzada
  otimiza a previsão e não a parcimônia
- **[03-01: Regressão logística](../03-classification/01-logistic-regression/)**,
  a sigmoide, a log loss, o gradiente descendente escrito por extenso, softmax em
  sete variedades de feijão de frequências bem diferentes, e por que justamente a
  classe *mais rara* acabou sendo a mais fácil
- **[03-02: k vizinhos mais próximos](../03-classification/02-k-nearest-neighbours/)**,
  o modelo que não treina nada; só o escalonamento vale **+0,205 de acurácia**,
  `k=1` tira um 1,000 perfeito e sem sentido no treino, e a maldição da
  dimensionalidade é medida em vez de afirmada
- **[03-03: Naive Bayes](../03-classification/03-naive-bayes/)**, por que uma
  suposição que todo mundo sabe ser falsa ainda funciona, e um desvio de treze
  pontos pelo padrão `var_smoothing` do scikit-learn, que destrói caladinho as
  variáveis de escala pequena
- **[03-05: Máquinas de vetores de suporte](../03-classification/05-support-vector-machines/)**,
  a margem, o que `C` controla de fato, e o truque do kernel mostrado erguendo
  dois anéis para uma terceira dimensão antes de explicar por que você nunca
  precisa construí-la; junto com o custo que ninguém menciona: 27× mais linhas
  custam **139× mais tempo**
- **[03-06: Árvores de decisão](../03-classification/06-decision-trees/)**, a
  busca pela divisão escrita do zero, a melhor pergunta de todo o conjunto de
  dados encontrada na aritmética, uma árvore legível de profundidade 3 e outra sem
  freio que chega à acurácia perfeita no treino enquanto *piora* nos dados
  separados
- **[04-02: Random Forests](../04-ensembles/02-random-forest/)**, a equação da
  variância que explica todo o projeto, a pontuação out-of-bag, duas medidas de
  importância de variáveis que discordam entre si, e um confronto direto com a
  regressão logística que a floresta **não vence**
- **[04-05: Gradient boosting](../04-ensembles/05-gradient-boosting/)**, por que
  ajustar o resíduo *é* gradiente descendente, o que a taxa de aprendizado cobra e
  o que ela devolve, e a diferença mais marcante em relação a uma floresta: a
  certa altura, acrescentar árvores deixa o boosting **pior**
- **[05-01: k-Means](../05-unsupervised/01-k-means/)**, o algoritmo de Lloyd do
  zero, o cotovelo e a silhueta apontando os dois para o número *errado* de
  clusters em dados cuja verdade é conhecida, e quatro modos de falhar, entre eles
  o k-means fatiando puro ruído em grupinhos bem-comportados sem titubear
- **[05-04: DBSCAN e HDBSCAN](../05-unsupervised/04-dbscan-and-hdbscan/)**,
  clusterização por densidade que encontra meias-luas onde o k-means não consegue,
  `eps` escolhido a partir do cotovelo da k-distância em vez de no chute, e a
  descoberta de que **nenhum valor de `eps` avisa que "estes dados não têm
  estrutura"**
- **[06-01: PCA](../06-dimensionality-reduction/01-principal-component-analysis/)**,
  a decomposição em autovalores feita na mão concordando com o scikit-learn até
  2,2×10⁻¹⁶, componentes que acabam significando "tamanho" e "forma", e a
  constatação honesta de que o PCA **nunca ganhou** de simplesmente manter todas
  as colunas
- **[07-01: O perceptron](../07-neural-networks/01-the-perceptron/)**, um
  neurônio, a regra de aprendizado que dispensa cálculo, e os quatro pontos do XOR
  que travaram as redes neurais por uma década
- **[07-02: MLP e retropropagação](../07-neural-networks/02-mlp-and-backpropagation/)**,
  cada gradiente deduzido à mão e **conferido numericamente até 1,9×10⁻¹⁰**, e
  depois uma rede funcionando só com NumPy
- **[07-03: A mesma rede em PyTorch](../07-neural-networks/03-the-same-net-in-pytorch/)**,
  autograd demonstrado em algo que dá para conferir à mão, o laço de treino de
  cinco linhas que você reaproveita para sempre, e os três erros que todo mundo
  comete uma vez
- **[08-02: Uma CNN, camada por camada](../08-computer-vision/02-a-cnn-layer-by-layer/)**,
  o compartilhamento de pesos explicado, e uma rede convolucional que ganha de uma
  densa em acurácia, em número de parâmetros **e** em velocidade; os filtros da
  sua primeira camada viram detectores de borda que ninguém pediu
- **[11-04: Q-learning](../11-reinforcement-learning/04-q-learning/)**, o Cliff
  Walking construído do zero sem depender do `gym`, a atualização de Bellman em
  uma linha, e o resultado clássico em que o Q-learning acha a política melhor
  enquanto o SARSA junta mais recompensa

Juntos, eles cobrem regressão supervisionada, classificação supervisionada,
ensembles, clusterização não supervisionada, redução de dimensionalidade e
aprendizado por reforço, e cada um deles está pronto e executado.

### O placar

Como todos os notebooks usam os mesmos conjuntos de dados, as comparações vão se
acumulando. No **UCI Dry Bean**, acurácia com validação cruzada de 5 partições:

| Método | Acurácia |
|---|---|
| SVM, kernel RBF | **0.9301** |
| Gradient boosting | 0.9271 |
| SVM, kernel linear | 0.9262 |
| Random forest | 0.9244 |
| Regressão logística | 0.9234 |
| k vizinhos mais próximos | 0.9231 |
| Rede neural (NumPy, do zero) | 0.9306 * |
| Naive Bayes | 0.8972 |
| Árvore de decisão | 0.8945 |

\* uma única partição separada em vez de 5 partições, então não é diretamente
comparável. No [notebook](../07-neural-networks/02-mlp-and-backpropagation/)
explico por que não reivindico isso como vitória.

Os números desta tabela aparecem exatamente como os notebooks os imprimem, com
ponto decimal; no texto desta página uso a vírgula, como é normal em português.

Nove métodos separados por menos de quatro centésimos, e uma reta ganha da
maioria deles. As medidas dos feijões são geometria suave e correlacionada, então
sobra pouca estrutura não linear para os modelos flexíveis explorarem.

Troque o conjunto de dados e a ordem muda. No **California Housing**, o gradient
boosting derruba o RMSE de 0,7263 da regressão linear para **0,4668**, uma melhora
de 36%, porque esse problema *é* cheio de interações.

**Qual método vence é uma propriedade dos seus dados, não um ranking de
algoritmos.** Esse é o fio que atravessa o livro inteiro, e vários notebooks aqui
relatam um resultado que eu esperava ao contrário e dizem isso sem rodeios.

---

## Como começar

```bash
git clone https://github.com/ELounissi/open-ml-notebooks
cd open-ml-notebooks
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Tudo roda **em minutos, na CPU de um laptop**. Uma GPU acelera os capítulos de
deep learning e nunca é obrigatória. Os conjuntos de dados pequenos estão no
repositório, então a maioria dos notebooks roda offline na hora.

**Começando do zero?** Comece pela [Parte 1, Fundamentos](../01-foundations/).
**Já sabe o básico?** Pule para o método que você precisa: cada notebook se sustenta sozinho.
**Preparando entrevistas?** A seção "quando ganha, quando perde" de cada notebook
foi escrita exatamente para essa conversa.

---

## Reaproveitar isto

O código é **MIT**. O texto e as figuras são **CC BY 4.0**. Leve para um curso,
uma palestra ou um grupo de estudos, com atribuição. Os conjuntos de dados mantêm
as próprias licenças, registradas um a um em
[`data/README.md`](../data/README.md).

Se isto te poupou tempo, **uma estrela ajuda outras pessoas a encontrarem o
livro**. Achou um erro? Abra uma issue: correções são muito bem-vindas,
principalmente as que mostram que alguma afirmação daqui está errada.

---

### Tópicos

`machine-learning` `deep-learning` `reinforcement-learning` `machine-learning-tutorial`
`jupyter-notebook` `python` `scikit-learn` `pytorch` `data-science` `neural-network`
`cnn` `rnn` `lstm` `transformer` `linear-regression` `logistic-regression`
`random-forest` `xgboost` `clustering` `pca` `learn-machine-learning`
`machine-learning-algorithms` `ml-tutorial` `data-science-tutorial` `education`

`#MachineLearning` `#DeepLearning` `#ReinforcementLearning` `#DataScience`
`#Python` `#ScikitLearn` `#PyTorch` `#LearnMachineLearning` `#MLTutorial`
`#JupyterNotebook` `#OpenSource` `#AI`

---

**Feito por Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)
