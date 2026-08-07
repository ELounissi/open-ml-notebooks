# Lehrplan

Sechsundachtzig Notebooks, zwölf Teile, je ein Verfahren. Lies sie der Reihe nach
oder spring direkt zu dem, das du brauchst: jedes Notebook steht für sich.

Jedes Notebook hat denselben fünfteiligen Aufbau; wenn du eines gelesen hast,
weißt du in allen anderen, wo du nachschauen musst:

1. **Die Idee**: was das Verfahren macht, in einfacher Sprache, mit einem Bild
2. **Die Mathematik**: nur so viel wie nötig, ausgeschrieben, ohne Auslassungen
3. **Von Hand**: eine minimale Implementierung in NumPy, damit nichts nach Magie aussieht
4. **In der Praxis**: die Fassung mit scikit-learn oder PyTorch, Zeile für Zeile kommentiert
5. **Wann es gewinnt, wann es verliert**: auf den Datensätzen des Buchs, mit Begründung

Jedes Notebook unten ist fertig: ausgeführt, mit seinen Diagrammen und seiner
eigenen Seite.

---

## Teil 1: Grundlagen

Das Vokabular, das alles andere benutzt. Fang hier an, wenn du neu bist.

| # | Notebook |
|---|---|
| 01-01 | [Was Machine Learning wirklich macht: der Ablauf von Anfang bis Ende](../01-foundations/01-what-machine-learning-does/) |
| 01-02 | [Training, Validierung, Test: warum du alle drei brauchst](../01-foundations/02-train-validation-test/) |
| 01-03 | [Overfitting und Underfitting, gesehen statt beschrieben](../01-foundations/03-overfitting-and-underfitting/) |
| 01-04 | [Kreuzvalidierung, und welche Variante wann](../01-foundations/04-cross-validation/) |
| 01-05 | [Metriken für Klassifikation: Accuracy, Precision, Recall, F1, ROC-AUC](../01-foundations/05-classification-metrics/) |
| 01-06 | [Metriken für Regression: MAE, RMSE, R², MAPE](../01-foundations/06-regression-metrics/) |
| 01-07 | [Skalierung von Features und Kodierung kategorialer Variablen](../01-foundations/07-feature-scaling-and-encoding/) |
| 01-08 | [Fehlende Werte: was zu tun ist und was es kostet](../01-foundations/08-missing-data/) |
| 01-09 | [Hyperparameter-Tuning: Grid, Random und Bayessche Suche](../01-foundations/09-hyperparameter-tuning/) |

## Teil 2: Regression

| # | Notebook |
|---|---|
| 02-01 | [Lineare Regression, von der Normalengleichung zu scikit-learn](../02-regression/01-linear-regression/) |
| 02-02 | [Gradientenabstieg, Schritt für Schritt beobachtet](../02-regression/02-gradient-descent/) |
| 02-03 | [Polynomiale Regression und das Bias-Varianz-Dilemma](../02-regression/03-polynomial-regression/) |
| 02-04 | [Ridge-Regression (L2)](../02-regression/04-ridge-regression/) |
| 02-05 | [Lasso-Regression (L1) und automatische Feature-Auswahl](../02-regression/05-lasso-regression/) |
| 02-06 | [Elastic Net](../02-regression/06-elastic-net/) |
| 02-07 | [Ausreißerrobuste Regression: Huber und RANSAC](../02-regression/07-outlier-resistant-regression/) |
| 02-08 | [Quantilsregression: einen Bereich vorhersagen statt einer Zahl](../02-regression/08-quantile-regression/) |
| 02-09 | [Poisson und andere verallgemeinerte lineare Modelle](../02-regression/09-generalised-linear-models/) |

## Teil 3: Klassifikation

| # | Notebook |
|---|---|
| 03-01 | [Logistische Regression](../03-classification/01-logistic-regression/) |
| 03-02 | [k-nächste Nachbarn](../03-classification/02-k-nearest-neighbours/) |
| 03-03 | [Naive Bayes](../03-classification/03-naive-bayes/) |
| 03-04 | [Lineare und quadratische Diskriminanzanalyse](../03-classification/04-discriminant-analysis/) |
| 03-05 | [Support Vector Machines und der Kernel-Trick](../03-classification/05-support-vector-machines/) |
| 03-06 | [Entscheidungsbäume, und wie ein Split gewählt wird](../03-classification/06-decision-trees/) |
| 03-07 | [Unausgewogene Klassen: Resampling, Gewichte und Schwellenwerte](../03-classification/07-imbalanced-classes/) |
| 03-08 | [Strategien für Multiclass und Multilabel](../03-classification/08-multiclass-and-multilabel/) |
| 03-09 | [Kalibrierung von Wahrscheinlichkeiten](../03-classification/09-probability-calibration/) |

## Teil 4: Ensembles

| # | Notebook |
|---|---|
| 04-01 | [Bagging, und warum Mitteln hilft](../04-ensembles/01-bagging/) |
| 04-02 | [Random Forests](../04-ensembles/02-random-forest/) |
| 04-03 | [Extra Trees](../04-ensembles/03-extra-trees/) |
| 04-04 | [AdaBoost](../04-ensembles/04-adaboost/) |
| 04-05 | [Gradient Boosting von Grund auf](../04-ensembles/05-gradient-boosting/) |
| 04-06 | [XGBoost, LightGBM und CatBoost im Vergleich](../04-ensembles/06-xgboost-lightgbm-catboost/) |
| 04-07 | [Stacking und Voting](../04-ensembles/07-stacking-and-voting/) |

## Teil 5: Unüberwachtes Lernen

| # | Notebook |
|---|---|
| 05-01 | [k-Means](../05-unsupervised/01-k-means/) |
| 05-02 | [k wählen: Ellenbogen, Silhouette, Gap-Statistik](../05-unsupervised/02-choosing-k/) |
| 05-03 | [Hierarchisches Clustering und Dendrogramme](../05-unsupervised/03-hierarchical-clustering/) |
| 05-04 | [DBSCAN und HDBSCAN](../05-unsupervised/04-dbscan-and-hdbscan/) |
| 05-05 | [Gaußsche Mischverteilungsmodelle](../05-unsupervised/05-gaussian-mixture-models/) |
| 05-06 | [Anomalieerkennung: Isolation Forest, One-Class SVM, LOF](../05-unsupervised/06-anomaly-detection/) |
| 05-07 | [Assoziationsregeln mit Apriori](../05-unsupervised/07-association-rules/) |

## Teil 6: Dimensionsreduktion

| # | Notebook |
|---|---|
| 06-01 | [Hauptkomponentenanalyse](../06-dimensionality-reduction/01-principal-component-analysis/) |
| 06-02 | [Kernel-PCA, ICA und NMF](../06-dimensionality-reduction/02-kernel-pca-ica-nmf/) |
| 06-03 | [t-SNE](../06-dimensionality-reduction/03-t-sne/) |
| 06-04 | [UMAP](../06-dimensionality-reduction/04-umap/) |
| 06-05 | [Feature-Auswahl: Filter, Wrapper, eingebettet](../06-dimensionality-reduction/05-feature-selection/) |

## Teil 7: Neuronale Netze

| # | Notebook |
|---|---|
| 07-01 | [Das Perzeptron](../07-neural-networks/01-the-perceptron/) |
| 07-02 | [Mehrschichtiges Perzeptron und Backpropagation, in NumPy](../07-neural-networks/02-mlp-and-backpropagation/) |
| 07-03 | [Dasselbe Netz in PyTorch](../07-neural-networks/03-the-same-net-in-pytorch/) |
| 07-04 | [Aktivierungsfunktionen und warum sie wichtig sind](../07-neural-networks/04-activation-functions/) |
| 07-05 | [Optimierer: SGD, Momentum, RMSProp, Adam](../07-neural-networks/05-optimisers/) |
| 07-06 | [Regularisierung: Dropout, Batch Norm, Weight Decay, Early Stopping](../07-neural-networks/06-regularisation/) |
| 07-07 | [Eine Trainingsschleife zum Wiederverwenden](../07-neural-networks/07-a-training-loop/) |

## Teil 8: Computer Vision

| # | Notebook |
|---|---|
| 08-01 | [Faltung, Pooling und was ein Filter lernt](../08-computer-vision/01-convolution-and-pooling/) |
| 08-02 | [Ein CNN auf Fashion-MNIST, Schicht für Schicht](../08-computer-vision/02-a-cnn-layer-by-layer/) |
| 08-03 | [Klassische Architekturen: LeNet, VGG, ResNet](../08-computer-vision/03-classic-architectures/) |
| 08-04 | [Data Augmentation](../08-computer-vision/04-data-augmentation/) |
| 08-05 | [Transfer Learning und Fine-Tuning](../08-computer-vision/05-transfer-learning/) |
| 08-06 | [Bildsegmentierung, eine Einführung](../08-computer-vision/06-image-segmentation/) |
| 08-07 | [Objekterkennung, eine Einführung](../08-computer-vision/07-object-detection/) |

## Teil 9: Sequenzen und Sprache

| # | Notebook |
|---|---|
| 09-01 | [Textvorverarbeitung, Bag of Words und TF-IDF](../09-sequences-and-language/01-text-preprocessing-and-tfidf/) |
| 09-02 | [Word Embeddings: Word2Vec und GloVe](../09-sequences-and-language/02-word-embeddings/) |
| 09-03 | [Rekurrente neuronale Netze](../09-sequences-and-language/03-recurrent-neural-networks/) |
| 09-04 | [LSTM](../09-sequences-and-language/04-lstm/) |
| 09-05 | [GRU, und wie es sich gegen LSTM schlägt](../09-sequences-and-language/05-gru/) |
| 09-06 | [Sequence to Sequence mit Attention](../09-sequences-and-language/06-seq2seq-with-attention/) |
| 09-07 | [Der Transformer, von Grund auf gebaut](../09-sequences-and-language/07-the-transformer/) |
| 09-08 | [Fine-Tuning eines vortrainierten Transformers](../09-sequences-and-language/08-fine-tuning/) |
| 09-09 | [Zeitreihenprognose: ARIMA gegen ML gegen Deep Learning](../09-sequences-and-language/09-time-series-forecasting/) |

## Teil 10: Generative Modelle

| # | Notebook |
|---|---|
| 10-01 | [Autoencoder](../10-generative-models/01-autoencoders/) |
| 10-02 | [Variational Autoencoder](../10-generative-models/02-variational-autoencoders/) |
| 10-03 | [Generative Adversarial Networks](../10-generative-models/03-generative-adversarial-networks/) |
| 10-04 | [Diffusionsmodelle, das kleinste lauffähige Beispiel](../10-generative-models/04-diffusion-models/) |

## Teil 11: Reinforcement Learning

| # | Notebook |
|---|---|
| 11-01 | [Der Aufbau beim Reinforcement Learning: Agenten, Zustände, Belohnungen](../11-reinforcement-learning/01-the-setup/) |
| 11-02 | [Multi-Armed Bandits](../11-reinforcement-learning/02-multi-armed-bandits/) |
| 11-03 | [Markov-Entscheidungsprozesse, Wert- und Strategieiteration](../11-reinforcement-learning/03-markov-decision-processes/) |
| 11-04 | [Q-Learning](../11-reinforcement-learning/04-q-learning/) |
| 11-05 | [SARSA, und worin es sich von Q-Learning unterscheidet](../11-reinforcement-learning/05-sarsa/) |
| 11-06 | [Deep Q-Networks](../11-reinforcement-learning/06-deep-q-networks/) |
| 11-07 | [Policy Gradients und REINFORCE](../11-reinforcement-learning/07-policy-gradients/) |
| 11-08 | [Actor-Critic und PPO](../11-reinforcement-learning/08-actor-critic-and-ppo/) |

## Teil 12: Alles zusammen

| # | Notebook |
|---|---|
| 12-01 | [**Die Ergebnistabelle**: jedes Verfahren auf jedem Datensatz des Buchs](../12-putting-it-together/01-the-scoreboard/) |
| 12-02 | [Modelle interpretieren: Permutationswichtigkeit, SHAP, LIME](../12-putting-it-together/02-interpreting-models/) |
| 12-03 | [Pipelines, und nie wieder Leakage](../12-putting-it-together/03-pipelines/) |
| 12-04 | [Ein Modell speichern, laden und ausliefern](../12-putting-it-together/04-saving-and-serving/) |
| 12-05 | [Die Fehler, die alle machen](../12-putting-it-together/05-common-mistakes/) |

---

## Die Datensätze des Buchs

Jedes Notebook benutzt denselben kleinen Satz, damit ein Vergleich über Kapitel
hinweg etwas bedeutet. Du siehst sofort, worin sich eine Support Vector Machine
und ein Random Forest unterscheiden, weil ihnen dieselbe Frage gestellt wurde.

| Datensatz | Aufgabe | Größe | Warum dieser |
|---|---|---|---|
| California Housing | Regression | 20.640 × 8 | Liegt scikit-learn bei, also läuft Kapitel eins ohne Download |
| Breast Cancer Wisconsin | binäre Klassifikation | 569 × 30 | Klein genug, dass jedes Verfahren sofort trainiert |
| **UCI Dry Bean** | Klassifikation mit 7 Klassen | 13.611 × 16 | 2020 veröffentlicht und in Tutorials kaum benutzt, was das Buch abseits der ausgetretenen Pfade hält |
| UCI Bike Sharing | Regression, Zeitreihe | 17.379 × 16 | Ein Datensatz, der für tabellarische Regression und für Sequenzmodelle taugt |
| Fashion-MNIST | Bildklassifikation | 70.000 × 28 × 28 | Dieselbe Form wie MNIST, um einiges schwerer, damit nicht alle CNN-Ergebnisse bei 99 % liegen |

Quellen, Lizenzen und Abrufdaten stehen in [`data/README.md`](../data/README.md).

## Die Ergebnistabelle

Teil 12 sammelt das Ergebnis jedes Verfahrens auf jedem Datensatz des Buchs in
einer Tabelle und erklärt dann das Muster. Manches davon ist vorhersehbar und
manches nicht: Gradient Boosting liegt bei Tabellendaten meistens vorn,
k-nächste Nachbarn zerfallen, sobald die Spalten mehr werden, ein lineares Modell
schlägt ein neuronales Netz, wenn es wenige Zeilen gibt. Der Sinn des Buchs ist,
dass du am Ende weißt *warum* und nicht nur *welches*.

---

Von **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

---

## Weiterverwenden, und dabei helfen, dass es Leser findet

Der Code steht unter der MIT-Lizenz. Kopier ihn, pass ihn an, nimm ihn in eigene
Projekte oder in deinen Unterricht, ohne Erlaubnis und ohne Namensnennung. Wenn
ein Kapitel dir einen Nachmittag spart, war es genau dafür da.

Wenn es dir nützt, **hilft ein Stern anderen dabei, es zu finden**, und anders
reist ein Buch wie dieses nicht. Korrekturen und Widerspruch sind in den Issues
willkommen, besonders wenn du etwas neu ausführst und ein anderes Ergebnis
bekommst.

---
