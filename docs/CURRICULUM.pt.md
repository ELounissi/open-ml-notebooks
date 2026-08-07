# Plano de estudos

Oitenta e seis notebooks, doze partes, um método em cada. Leia na ordem ou vá
direto ao que você precisa: cada notebook se sustenta sozinho.

Todos os notebooks seguem o mesmo formato de cinco partes, então depois de ler um
você já sabe onde procurar em todos os outros:

1. **A ideia**: o que o método faz, em linguagem simples, com uma figura
2. **A matemática**: só o que você precisa, escrita, sem pular etapas
3. **Do zero**: uma implementação mínima em NumPy, para que nada seja mágica
4. **Na prática**: a versão em scikit-learn ou PyTorch, comentada linha a linha
5. **Quando ganha, quando perde**: nos conjuntos de dados do livro, com o motivo

Todos os notebooks abaixo estão prontos: executados, com seus gráficos e sua
própria página.

---

## Parte 1: Fundamentos

O vocabulário que todo o resto usa. Comece aqui se você é novo no assunto.

| # | Notebook |
|---|---|
| 01-01 | [O que o aprendizado de máquina faz de verdade: o fluxo de trabalho de ponta a ponta](../01-foundations/01-what-machine-learning-does/) |
| 01-02 | [Treino, validação e teste: por que você precisa dos três](../01-foundations/02-train-validation-test/) |
| 01-03 | [Overfitting e underfitting, vistos em vez de descritos](../01-foundations/03-overfitting-and-underfitting/) |
| 01-04 | [Validação cruzada, e qual usar em cada situação](../01-foundations/04-cross-validation/) |
| 01-05 | [Métricas de classificação: acurácia, precisão, recall, F1, ROC-AUC](../01-foundations/05-classification-metrics/) |
| 01-06 | [Métricas de regressão: MAE, RMSE, R², MAPE](../01-foundations/06-regression-metrics/) |
| 01-07 | [Escalonamento de variáveis e codificação de variáveis categóricas](../01-foundations/07-feature-scaling-and-encoding/) |
| 01-08 | [Dados faltantes: o que fazer e quanto custa](../01-foundations/08-missing-data/) |
| 01-09 | [Ajuste de hiperparâmetros: busca em grade, aleatória e bayesiana](../01-foundations/09-hyperparameter-tuning/) |

## Parte 2: Regressão

| # | Notebook |
|---|---|
| 02-01 | [Regressão linear, da equação normal ao scikit-learn](../02-regression/01-linear-regression/) |
| 02-02 | [Gradiente descendente, observado passo a passo](../02-regression/02-gradient-descent/) |
| 02-03 | [Regressão polinomial e o equilíbrio entre viés e variância](../02-regression/03-polynomial-regression/) |
| 02-04 | [Regressão Ridge (L2)](../02-regression/04-ridge-regression/) |
| 02-05 | [Regressão Lasso (L1) e seleção automática de variáveis](../02-regression/05-lasso-regression/) |
| 02-06 | [Elastic Net](../02-regression/06-elastic-net/) |
| 02-07 | [Regressão resistente a outliers: Huber e RANSAC](../02-regression/07-outlier-resistant-regression/) |
| 02-08 | [Regressão quantílica: prever uma faixa, não um número](../02-regression/08-quantile-regression/) |
| 02-09 | [Poisson e outros modelos lineares generalizados](../02-regression/09-generalised-linear-models/) |

## Parte 3: Classificação

| # | Notebook |
|---|---|
| 03-01 | [Regressão logística](../03-classification/01-logistic-regression/) |
| 03-02 | [k vizinhos mais próximos](../03-classification/02-k-nearest-neighbours/) |
| 03-03 | [Naive Bayes](../03-classification/03-naive-bayes/) |
| 03-04 | [Análise discriminante linear e quadrática](../03-classification/04-discriminant-analysis/) |
| 03-05 | [Máquinas de vetores de suporte e o truque do kernel](../03-classification/05-support-vector-machines/) |
| 03-06 | [Árvores de decisão, e como uma divisão é escolhida](../03-classification/06-decision-trees/) |
| 03-07 | [Classes desbalanceadas: reamostragem, pesos e limiares](../03-classification/07-imbalanced-classes/) |
| 03-08 | [Estratégias multiclasse e multirrótulo](../03-classification/08-multiclass-and-multilabel/) |
| 03-09 | [Calibração de probabilidades](../03-classification/09-probability-calibration/) |

## Parte 4: Ensembles

| # | Notebook |
|---|---|
| 04-01 | [Bagging, e por que fazer a média ajuda](../04-ensembles/01-bagging/) |
| 04-02 | [Random Forests](../04-ensembles/02-random-forest/) |
| 04-03 | [Extra Trees](../04-ensembles/03-extra-trees/) |
| 04-04 | [AdaBoost](../04-ensembles/04-adaboost/) |
| 04-05 | [Gradient boosting a partir dos primeiros princípios](../04-ensembles/05-gradient-boosting/) |
| 04-06 | [XGBoost, LightGBM e CatBoost comparados](../04-ensembles/06-xgboost-lightgbm-catboost/) |
| 04-07 | [Stacking e voting](../04-ensembles/07-stacking-and-voting/) |

## Parte 5: Aprendizado não supervisionado

| # | Notebook |
|---|---|
| 05-01 | [k-Means](../05-unsupervised/01-k-means/) |
| 05-02 | [Escolhendo k: cotovelo, silhueta, estatística gap](../05-unsupervised/02-choosing-k/) |
| 05-03 | [Clusterização hierárquica e dendrogramas](../05-unsupervised/03-hierarchical-clustering/) |
| 05-04 | [DBSCAN e HDBSCAN](../05-unsupervised/04-dbscan-and-hdbscan/) |
| 05-05 | [Modelos de mistura de gaussianas](../05-unsupervised/05-gaussian-mixture-models/) |
| 05-06 | [Detecção de anomalias: Isolation Forest, One-Class SVM, LOF](../05-unsupervised/06-anomaly-detection/) |
| 05-07 | [Regras de associação com Apriori](../05-unsupervised/07-association-rules/) |

## Parte 6: Redução de dimensionalidade

| # | Notebook |
|---|---|
| 06-01 | [Análise de componentes principais](../06-dimensionality-reduction/01-principal-component-analysis/) |
| 06-02 | [Kernel PCA, ICA e NMF](../06-dimensionality-reduction/02-kernel-pca-ica-nmf/) |
| 06-03 | [t-SNE](../06-dimensionality-reduction/03-t-sne/) |
| 06-04 | [UMAP](../06-dimensionality-reduction/04-umap/) |
| 06-05 | [Seleção de variáveis: filtro, wrapper e embutida](../06-dimensionality-reduction/05-feature-selection/) |

## Parte 7: Redes neurais

| # | Notebook |
|---|---|
| 07-01 | [O perceptron](../07-neural-networks/01-the-perceptron/) |
| 07-02 | [Perceptron multicamadas e retropropagação, em NumPy](../07-neural-networks/02-mlp-and-backpropagation/) |
| 07-03 | [A mesma rede em PyTorch](../07-neural-networks/03-the-same-net-in-pytorch/) |
| 07-04 | [Funções de ativação e por que elas importam](../07-neural-networks/04-activation-functions/) |
| 07-05 | [Otimizadores: SGD, Momentum, RMSProp, Adam](../07-neural-networks/05-optimisers/) |
| 07-06 | [Regularização: dropout, batch norm, weight decay, early stopping](../07-neural-networks/06-regularisation/) |
| 07-07 | [Um laço de treino para reaproveitar](../07-neural-networks/07-a-training-loop/) |

## Parte 8: Visão computacional

| # | Notebook |
|---|---|
| 08-01 | [Convolução, pooling e o que um filtro aprende](../08-computer-vision/01-convolution-and-pooling/) |
| 08-02 | [Uma CNN no Fashion-MNIST, camada por camada](../08-computer-vision/02-a-cnn-layer-by-layer/) |
| 08-03 | [Arquiteturas clássicas: LeNet, VGG, ResNet](../08-computer-vision/03-classic-architectures/) |
| 08-04 | [Aumento de dados](../08-computer-vision/04-data-augmentation/) |
| 08-05 | [Transfer learning e fine-tuning](../08-computer-vision/05-transfer-learning/) |
| 08-06 | [Segmentação de imagens, uma introdução](../08-computer-vision/06-image-segmentation/) |
| 08-07 | [Detecção de objetos, uma introdução](../08-computer-vision/07-object-detection/) |

## Parte 9: Sequências e linguagem

| # | Notebook |
|---|---|
| 09-01 | [Pré-processamento de texto, bag of words e TF-IDF](../09-sequences-and-language/01-text-preprocessing-and-tfidf/) |
| 09-02 | [Word embeddings: Word2Vec e GloVe](../09-sequences-and-language/02-word-embeddings/) |
| 09-03 | [Redes neurais recorrentes](../09-sequences-and-language/03-recurrent-neural-networks/) |
| 09-04 | [LSTM](../09-sequences-and-language/04-lstm/) |
| 09-05 | [GRU, e como ela se compara à LSTM](../09-sequences-and-language/05-gru/) |
| 09-06 | [Sequência a sequência com atenção](../09-sequences-and-language/06-seq2seq-with-attention/) |
| 09-07 | [O Transformer, construído do zero](../09-sequences-and-language/07-the-transformer/) |
| 09-08 | [Fine-tuning de um transformer pré-treinado](../09-sequences-and-language/08-fine-tuning/) |
| 09-09 | [Previsão de séries temporais: ARIMA contra ML contra deep learning](../09-sequences-and-language/09-time-series-forecasting/) |

## Parte 10: Modelos generativos

| # | Notebook |
|---|---|
| 10-01 | [Autoencoders](../10-generative-models/01-autoencoders/) |
| 10-02 | [Autoencoders variacionais](../10-generative-models/02-variational-autoencoders/) |
| 10-03 | [Redes adversárias generativas](../10-generative-models/03-generative-adversarial-networks/) |
| 10-04 | [Modelos de difusão, o menor exemplo que funciona](../10-generative-models/04-diffusion-models/) |

## Parte 11: Aprendizado por reforço

| # | Notebook |
|---|---|
| 11-01 | [O cenário do aprendizado por reforço: agentes, estados, recompensas](../11-reinforcement-learning/01-the-setup/) |
| 11-02 | [Bandidos de múltiplos braços](../11-reinforcement-learning/02-multi-armed-bandits/) |
| 11-03 | [Processos de decisão de Markov, iteração de valor e de política](../11-reinforcement-learning/03-markov-decision-processes/) |
| 11-04 | [Q-learning](../11-reinforcement-learning/04-q-learning/) |
| 11-05 | [SARSA, e em que ele difere do Q-learning](../11-reinforcement-learning/05-sarsa/) |
| 11-06 | [Deep Q-Networks](../11-reinforcement-learning/06-deep-q-networks/) |
| 11-07 | [Gradientes de política e REINFORCE](../11-reinforcement-learning/07-policy-gradients/) |
| 11-08 | [Ator-crítico e PPO](../11-reinforcement-learning/08-actor-critic-and-ppo/) |

## Parte 12: Juntando tudo

| # | Notebook |
|---|---|
| 12-01 | [**O placar**: cada método em cada conjunto de dados do livro](../12-putting-it-together/01-the-scoreboard/) |
| 12-02 | [Interpretando modelos: importância por permutação, SHAP, LIME](../12-putting-it-together/02-interpreting-models/) |
| 12-03 | [Pipelines, e nunca mais vazar dados](../12-putting-it-together/03-pipelines/) |
| 12-04 | [Salvar, carregar e servir um modelo](../12-putting-it-together/04-saving-and-serving/) |
| 12-05 | [Os erros que todo mundo comete](../12-putting-it-together/05-common-mistakes/) |

---

## Os conjuntos de dados do livro

Todos os notebooks usam o mesmo grupo pequeno, então comparar entre capítulos
quer dizer alguma coisa. Dá para ver na hora como uma máquina de vetores de
suporte e um Random Forest diferem, porque a pergunta feita a eles foi a mesma.

| Conjunto de dados | Tarefa | Tamanho | Por que este |
|---|---|---|---|
| California Housing | regressão | 20.640 × 8 | Vem junto com o scikit-learn, então o capítulo um roda sem baixar nada |
| Breast Cancer Wisconsin | classificação binária | 569 × 30 | Pequeno o bastante para qualquer método treinar na hora |
| **UCI Dry Bean** | classificação em 7 classes | 13.611 × 16 | Publicado em 2020 e quase não usado em tutoriais, o que mantém o livro fora do caminho batido |
| UCI Bike Sharing | regressão, séries temporais | 17.379 × 16 | Um conjunto só que funciona tanto para regressão tabular quanto para modelos de sequência |
| Fashion-MNIST | classificação de imagens | 70.000 × 28 × 28 | O mesmo formato do MNIST, várias vezes mais difícil, então os resultados das CNNs não ficam todos em 99% |

As fontes, as licenças e as datas de coleta estão em
[`data/README.md`](../data/README.md).

## O placar

A parte 12 reúne em uma tabela o resultado de cada método em cada conjunto de
dados do livro e depois explica o padrão. Parte disso é previsível e parte não:
gradient boosting costuma liderar em dados tabulares, os k vizinhos mais próximos
desandam quando as colunas se multiplicam, um modelo linear ganha de uma rede
neural quando há poucas linhas. O sentido do livro é você terminar sabendo *por
quê*, e não só *qual*.

---

Feito por **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

---

## Usar isto, e ajudar a fazer chegar às pessoas

O código está sob licença MIT. Copie, adapte, use nos seus projetos ou nas suas
aulas, sem pedir permissão e sem precisar dar crédito. Se um capítulo te poupar
uma tarde, era para isso mesmo.

Se for útil para você, **uma estrela ajuda outras pessoas a encontrarem o
livro**, que é o único jeito de um livro assim circular. Correções e discordâncias
são bem-vindas nas issues, principalmente se você reexecutar algo e obtiver outra
resposta.

---
