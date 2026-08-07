# Plan de estudios

Ochenta y seis notebooks, doce partes, un método en cada uno. Léelos en orden o
ve directo al que necesites: cada notebook se sostiene solo.

Todos los notebooks tienen la misma forma en cinco partes, así que en cuanto has
leído uno ya sabes dónde mirar en los demás:

1. **La idea**: qué hace el método, en lenguaje llano, con un dibujo
2. **Las matemáticas**: solo lo que necesitas, escrito, sin saltarse pasos
3. **Desde cero**: una implementación mínima en NumPy, para que nada sea magia
4. **En la práctica**: la versión de scikit-learn o PyTorch, comentada línea a línea
5. **Cuándo gana y cuándo pierde**: sobre los conjuntos de datos del libro, con el motivo

Todos los notebooks de abajo están terminados: ejecutados, con sus gráficas y con
su propia página.

---

## Parte 1: Fundamentos

El vocabulario que usa todo lo demás. Empieza aquí si vienes de cero.

| # | Notebook |
|---|---|
| 01-01 | [Qué hace realmente el aprendizaje automático: el flujo de trabajo de principio a fin](../01-foundations/01-what-machine-learning-does/) |
| 01-02 | [Entrenamiento, validación y test: por qué necesitas los tres](../01-foundations/02-train-validation-test/) |
| 01-03 | [Overfitting y underfitting, vistos en lugar de descritos](../01-foundations/03-overfitting-and-underfitting/) |
| 01-04 | [Validación cruzada, y cuál usar en cada caso](../01-foundations/04-cross-validation/) |
| 01-05 | [Métricas de clasificación: exactitud, precisión, recall, F1, ROC-AUC](../01-foundations/05-classification-metrics/) |
| 01-06 | [Métricas de regresión: MAE, RMSE, R², MAPE](../01-foundations/06-regression-metrics/) |
| 01-07 | [Escalado de variables y codificación de variables categóricas](../01-foundations/07-feature-scaling-and-encoding/) |
| 01-08 | [Datos faltantes: qué hacer y qué cuesta](../01-foundations/08-missing-data/) |
| 01-09 | [Ajuste de hiperparámetros: búsqueda en rejilla, aleatoria y bayesiana](../01-foundations/09-hyperparameter-tuning/) |

## Parte 2: Regresión

| # | Notebook |
|---|---|
| 02-01 | [Regresión lineal, de la ecuación normal a scikit-learn](../02-regression/01-linear-regression/) |
| 02-02 | [Descenso de gradiente, observado paso a paso](../02-regression/02-gradient-descent/) |
| 02-03 | [Regresión polinómica y el compromiso entre sesgo y varianza](../02-regression/03-polynomial-regression/) |
| 02-04 | [Regresión Ridge (L2)](../02-regression/04-ridge-regression/) |
| 02-05 | [Regresión Lasso (L1) y selección automática de variables](../02-regression/05-lasso-regression/) |
| 02-06 | [Elastic Net](../02-regression/06-elastic-net/) |
| 02-07 | [Regresión resistente a valores atípicos: Huber y RANSAC](../02-regression/07-outlier-resistant-regression/) |
| 02-08 | [Regresión cuantílica: predecir un rango, no un número](../02-regression/08-quantile-regression/) |
| 02-09 | [Poisson y otros modelos lineales generalizados](../02-regression/09-generalised-linear-models/) |

## Parte 3: Clasificación

| # | Notebook |
|---|---|
| 03-01 | [Regresión logística](../03-classification/01-logistic-regression/) |
| 03-02 | [k vecinos más cercanos](../03-classification/02-k-nearest-neighbours/) |
| 03-03 | [Naive Bayes](../03-classification/03-naive-bayes/) |
| 03-04 | [Análisis discriminante lineal y cuadrático](../03-classification/04-discriminant-analysis/) |
| 03-05 | [Máquinas de vectores de soporte y el truco del kernel](../03-classification/05-support-vector-machines/) |
| 03-06 | [Árboles de decisión, y cómo se elige una división](../03-classification/06-decision-trees/) |
| 03-07 | [Clases desequilibradas: remuestreo, pesos y umbrales](../03-classification/07-imbalanced-classes/) |
| 03-08 | [Estrategias multiclase y multietiqueta](../03-classification/08-multiclass-and-multilabel/) |
| 03-09 | [Calibración de probabilidades](../03-classification/09-probability-calibration/) |

## Parte 4: Ensembles

| # | Notebook |
|---|---|
| 04-01 | [Bagging, y por qué promediar ayuda](../04-ensembles/01-bagging/) |
| 04-02 | [Random Forests](../04-ensembles/02-random-forest/) |
| 04-03 | [Extra Trees](../04-ensembles/03-extra-trees/) |
| 04-04 | [AdaBoost](../04-ensembles/04-adaboost/) |
| 04-05 | [Gradient boosting desde los primeros principios](../04-ensembles/05-gradient-boosting/) |
| 04-06 | [XGBoost, LightGBM y CatBoost comparados](../04-ensembles/06-xgboost-lightgbm-catboost/) |
| 04-07 | [Stacking y voting](../04-ensembles/07-stacking-and-voting/) |

## Parte 5: Aprendizaje no supervisado

| # | Notebook |
|---|---|
| 05-01 | [k-Means](../05-unsupervised/01-k-means/) |
| 05-02 | [Elegir k: codo, silueta, estadístico gap](../05-unsupervised/02-choosing-k/) |
| 05-03 | [Clustering jerárquico y dendrogramas](../05-unsupervised/03-hierarchical-clustering/) |
| 05-04 | [DBSCAN y HDBSCAN](../05-unsupervised/04-dbscan-and-hdbscan/) |
| 05-05 | [Modelos de mezcla de gaussianas](../05-unsupervised/05-gaussian-mixture-models/) |
| 05-06 | [Detección de anomalías: Isolation Forest, One-Class SVM, LOF](../05-unsupervised/06-anomaly-detection/) |
| 05-07 | [Reglas de asociación con Apriori](../05-unsupervised/07-association-rules/) |

## Parte 6: Reducción de dimensionalidad

| # | Notebook |
|---|---|
| 06-01 | [Análisis de componentes principales](../06-dimensionality-reduction/01-principal-component-analysis/) |
| 06-02 | [Kernel PCA, ICA y NMF](../06-dimensionality-reduction/02-kernel-pca-ica-nmf/) |
| 06-03 | [t-SNE](../06-dimensionality-reduction/03-t-sne/) |
| 06-04 | [UMAP](../06-dimensionality-reduction/04-umap/) |
| 06-05 | [Selección de variables: filtro, wrapper y embebida](../06-dimensionality-reduction/05-feature-selection/) |

## Parte 7: Redes neuronales

| # | Notebook |
|---|---|
| 07-01 | [El perceptrón](../07-neural-networks/01-the-perceptron/) |
| 07-02 | [Perceptrón multicapa y retropropagación, en NumPy](../07-neural-networks/02-mlp-and-backpropagation/) |
| 07-03 | [La misma red en PyTorch](../07-neural-networks/03-the-same-net-in-pytorch/) |
| 07-04 | [Funciones de activación y por qué importan](../07-neural-networks/04-activation-functions/) |
| 07-05 | [Optimizadores: SGD, Momentum, RMSProp, Adam](../07-neural-networks/05-optimisers/) |
| 07-06 | [Regularización: dropout, batch norm, weight decay, early stopping](../07-neural-networks/06-regularisation/) |
| 07-07 | [Un bucle de entrenamiento que puedes reutilizar](../07-neural-networks/07-a-training-loop/) |

## Parte 8: Visión artificial

| # | Notebook |
|---|---|
| 08-01 | [Convolución, pooling y qué aprende un filtro](../08-computer-vision/01-convolution-and-pooling/) |
| 08-02 | [Una CNN sobre Fashion-MNIST, capa por capa](../08-computer-vision/02-a-cnn-layer-by-layer/) |
| 08-03 | [Arquitecturas clásicas: LeNet, VGG, ResNet](../08-computer-vision/03-classic-architectures/) |
| 08-04 | [Aumento de datos](../08-computer-vision/04-data-augmentation/) |
| 08-05 | [Transfer learning y ajuste fino](../08-computer-vision/05-transfer-learning/) |
| 08-06 | [Segmentación de imágenes, una introducción](../08-computer-vision/06-image-segmentation/) |
| 08-07 | [Detección de objetos, una introducción](../08-computer-vision/07-object-detection/) |

## Parte 9: Secuencias y lenguaje

| # | Notebook |
|---|---|
| 09-01 | [Preprocesado de texto, bolsa de palabras y TF-IDF](../09-sequences-and-language/01-text-preprocessing-and-tfidf/) |
| 09-02 | [Word embeddings: Word2Vec y GloVe](../09-sequences-and-language/02-word-embeddings/) |
| 09-03 | [Redes neuronales recurrentes](../09-sequences-and-language/03-recurrent-neural-networks/) |
| 09-04 | [LSTM](../09-sequences-and-language/04-lstm/) |
| 09-05 | [GRU, y cómo se compara con LSTM](../09-sequences-and-language/05-gru/) |
| 09-06 | [Secuencia a secuencia con atención](../09-sequences-and-language/06-seq2seq-with-attention/) |
| 09-07 | [El Transformer, construido desde cero](../09-sequences-and-language/07-the-transformer/) |
| 09-08 | [Ajuste fino de un transformer preentrenado](../09-sequences-and-language/08-fine-tuning/) |
| 09-09 | [Predicción de series temporales: ARIMA frente a ML frente a deep learning](../09-sequences-and-language/09-time-series-forecasting/) |

## Parte 10: Modelos generativos

| # | Notebook |
|---|---|
| 10-01 | [Autoencoders](../10-generative-models/01-autoencoders/) |
| 10-02 | [Autoencoders variacionales](../10-generative-models/02-variational-autoencoders/) |
| 10-03 | [Redes generativas antagónicas](../10-generative-models/03-generative-adversarial-networks/) |
| 10-04 | [Modelos de difusión, el ejemplo más pequeño que funciona](../10-generative-models/04-diffusion-models/) |

## Parte 11: Aprendizaje por refuerzo

| # | Notebook |
|---|---|
| 11-01 | [El montaje del aprendizaje por refuerzo: agentes, estados, recompensas](../11-reinforcement-learning/01-the-setup/) |
| 11-02 | [Bandidos multibrazo](../11-reinforcement-learning/02-multi-armed-bandits/) |
| 11-03 | [Procesos de decisión de Markov, iteración de valor y de política](../11-reinforcement-learning/03-markov-decision-processes/) |
| 11-04 | [Q-learning](../11-reinforcement-learning/04-q-learning/) |
| 11-05 | [SARSA, y en qué se diferencia de Q-learning](../11-reinforcement-learning/05-sarsa/) |
| 11-06 | [Deep Q-Networks](../11-reinforcement-learning/06-deep-q-networks/) |
| 11-07 | [Gradientes de política y REINFORCE](../11-reinforcement-learning/07-policy-gradients/) |
| 11-08 | [Actor-crítico y PPO](../11-reinforcement-learning/08-actor-critic-and-ppo/) |

## Parte 12: Todo junto

| # | Notebook |
|---|---|
| 12-01 | [**El marcador**: cada método sobre cada conjunto de datos del libro](../12-putting-it-together/01-the-scoreboard/) |
| 12-02 | [Interpretar modelos: importancia por permutación, SHAP, LIME](../12-putting-it-together/02-interpreting-models/) |
| 12-03 | [Pipelines, y no volver a tener fugas](../12-putting-it-together/03-pipelines/) |
| 12-04 | [Guardar, cargar y servir un modelo](../12-putting-it-together/04-saving-and-serving/) |
| 12-05 | [Los errores que comete todo el mundo](../12-putting-it-together/05-common-mistakes/) |

---

## Los conjuntos de datos del libro

Todos los notebooks usan el mismo grupo pequeño, así que comparar entre capítulos
significa algo. Ves de inmediato en qué se diferencian una máquina de vectores de
soporte y un Random Forest, porque se les hizo la misma pregunta.

| Conjunto de datos | Tarea | Tamaño | Por qué este |
|---|---|---|---|
| California Housing | regresión | 20.640 × 8 | Viene con scikit-learn, así que el capítulo uno funciona sin descargar nada |
| Breast Cancer Wisconsin | clasificación binaria | 569 × 30 | Lo bastante pequeño para que cualquier método entrene al instante |
| **UCI Dry Bean** | clasificación en 7 clases | 13.611 × 16 | Publicado en 2020 y apenas usado en tutoriales, lo que mantiene el libro fuera del camino trillado |
| UCI Bike Sharing | regresión, series temporales | 17.379 × 16 | Un solo conjunto que vale para regresión tabular y para modelos de secuencias |
| Fashion-MNIST | clasificación de imágenes | 70.000 × 28 × 28 | La misma forma que MNIST, varias veces más difícil, así que los resultados de las CNN no son todos del 99 % |

Las fuentes, las licencias y las fechas de descarga están en
[`data/README.md`](../data/README.md).

## El marcador

La parte 12 reúne en una tabla el resultado de cada método sobre cada conjunto de
datos del libro y después explica el patrón. Parte de eso es previsible y parte
no: gradient boosting suele ir por delante en datos tabulares, los k vecinos más
cercanos se deshacen cuando se multiplican las columnas, un modelo lineal le gana
a una red neuronal cuando hay pocas filas. El sentido del libro es que lo
termines sabiendo *por qué*, y no solo *cuál*.

---

Hecho por **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

---

## Usar esto, y ayudar a que llegue a más gente

El código tiene licencia MIT. Cópialo, adáptalo, mételo en tus propios proyectos
o en tus clases, sin pedir permiso y sin citar la fuente. Si un capítulo te
ahorra una tarde, para eso estaba.

Si te resulta útil, **una estrella ayuda a que otras personas lo encuentren**,
que es la única forma en que viaja un libro como este. Las correcciones y los
desacuerdos son bienvenidos en los issues, sobre todo si vuelves a ejecutar algo
y te sale otra cosa.

---
