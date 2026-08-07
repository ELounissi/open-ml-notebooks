# Programme

Quatre-vingt-six notebooks, douze parties, une méthode chacun. Lisez-les dans
l'ordre ou allez droit à celui qu'il vous faut : chaque notebook se lit seul.

Chaque notebook suit la même structure en cinq temps, donc une fois que vous en
avez lu un, vous savez où regarder dans tous les autres :

1. **L'idée** : ce que fait la méthode, en langage simple, avec une image
2. **Les maths** : seulement ce qui sert, écrit noir sur blanc, sans esquive
3. **De zéro** : une implémentation minimale en NumPy, pour que rien ne reste magique
4. **En pratique** : la version scikit-learn ou PyTorch, annotée ligne à ligne
5. **Quand ça gagne, quand ça perd** : sur les jeux de données maison, avec la raison

Tous les notebooks ci-dessous sont terminés : exécutés, avec leurs graphiques et
leur propre page.

---

## Partie 1 : Fondamentaux

Le vocabulaire dont tout le reste se sert. Commencez ici si vous débutez.

| # | Notebook |
|---|---|
| 01-01 | [Ce que fait vraiment le machine learning : le déroulé complet d'un projet](../01-foundations/01-what-machine-learning-does/) |
| 01-02 | [Entraînement, validation, test : pourquoi il vous faut les trois](../01-foundations/02-train-validation-test/) |
| 01-03 | [Surapprentissage et sous-apprentissage, vus plutôt que décrits](../01-foundations/03-overfitting-and-underfitting/) |
| 01-04 | [La validation croisée, et quelle variante utiliser quand](../01-foundations/04-cross-validation/) |
| 01-05 | [Métriques de classification : accuracy, précision, rappel, F1, ROC-AUC](../01-foundations/05-classification-metrics/) |
| 01-06 | [Métriques de régression : MAE, RMSE, R², MAPE](../01-foundations/06-regression-metrics/) |
| 01-07 | [Mise à l'échelle des variables et encodage des variables catégorielles](../01-foundations/07-feature-scaling-and-encoding/) |
| 01-08 | [Données manquantes : quoi faire, et ce que ça coûte](../01-foundations/08-missing-data/) |
| 01-09 | [Réglage des hyperparamètres : grille, aléatoire et recherche bayésienne](../01-foundations/09-hyperparameter-tuning/) |

## Partie 2 : Régression

| # | Notebook |
|---|---|
| 02-01 | [Régression linéaire, de l'équation normale à scikit-learn](../02-regression/01-linear-regression/) |
| 02-02 | [La descente de gradient, observée pas à pas](../02-regression/02-gradient-descent/) |
| 02-03 | [Régression polynomiale et compromis biais-variance](../02-regression/03-polynomial-regression/) |
| 02-04 | [Régression Ridge (L2)](../02-regression/04-ridge-regression/) |
| 02-05 | [Régression Lasso (L1) et sélection automatique de variables](../02-regression/05-lasso-regression/) |
| 02-06 | [Elastic Net](../02-regression/06-elastic-net/) |
| 02-07 | [Régression robuste aux valeurs aberrantes : Huber et RANSAC](../02-regression/07-outlier-resistant-regression/) |
| 02-08 | [Régression quantile : prédire un intervalle, pas un nombre](../02-regression/08-quantile-regression/) |
| 02-09 | [Poisson et les autres modèles linéaires généralisés](../02-regression/09-generalised-linear-models/) |

## Partie 3 : Classification

| # | Notebook |
|---|---|
| 03-01 | [Régression logistique](../03-classification/01-logistic-regression/) |
| 03-02 | [k plus proches voisins](../03-classification/02-k-nearest-neighbours/) |
| 03-03 | [Naive Bayes](../03-classification/03-naive-bayes/) |
| 03-04 | [Analyse discriminante linéaire et quadratique](../03-classification/04-discriminant-analysis/) |
| 03-05 | [Machines à vecteurs de support et l'astuce du noyau](../03-classification/05-support-vector-machines/) |
| 03-06 | [Arbres de décision, et comment se choisit une coupure](../03-classification/06-decision-trees/) |
| 03-07 | [Classes déséquilibrées : rééchantillonnage, pondération et seuils](../03-classification/07-imbalanced-classes/) |
| 03-08 | [Stratégies multiclasses et multilabels](../03-classification/08-multiclass-and-multilabel/) |
| 03-09 | [Calibration des probabilités](../03-classification/09-probability-calibration/) |

## Partie 4 : Ensembles

| # | Notebook |
|---|---|
| 04-01 | [Le bagging, et pourquoi moyenner aide](../04-ensembles/01-bagging/) |
| 04-02 | [Forêts aléatoires](../04-ensembles/02-random-forest/) |
| 04-03 | [Extra Trees](../04-ensembles/03-extra-trees/) |
| 04-04 | [AdaBoost](../04-ensembles/04-adaboost/) |
| 04-05 | [Le gradient boosting repris depuis la base](../04-ensembles/05-gradient-boosting/) |
| 04-06 | [XGBoost, LightGBM et CatBoost comparés](../04-ensembles/06-xgboost-lightgbm-catboost/) |
| 04-07 | [Stacking et vote](../04-ensembles/07-stacking-and-voting/) |

## Partie 5 : Apprentissage non supervisé

| # | Notebook |
|---|---|
| 05-01 | [k-Means](../05-unsupervised/01-k-means/) |
| 05-02 | [Choisir k : coude, silhouette, statistique de gap](../05-unsupervised/02-choosing-k/) |
| 05-03 | [Clustering hiérarchique et dendrogrammes](../05-unsupervised/03-hierarchical-clustering/) |
| 05-04 | [DBSCAN et HDBSCAN](../05-unsupervised/04-dbscan-and-hdbscan/) |
| 05-05 | [Modèles de mélange gaussien](../05-unsupervised/05-gaussian-mixture-models/) |
| 05-06 | [Détection d'anomalies : Isolation Forest, One-Class SVM, LOF](../05-unsupervised/06-anomaly-detection/) |
| 05-07 | [Règles d'association avec Apriori](../05-unsupervised/07-association-rules/) |

## Partie 6 : Réduction de dimension

| # | Notebook |
|---|---|
| 06-01 | [Analyse en composantes principales](../06-dimensionality-reduction/01-principal-component-analysis/) |
| 06-02 | [Kernel PCA, ICA et NMF](../06-dimensionality-reduction/02-kernel-pca-ica-nmf/) |
| 06-03 | [t-SNE](../06-dimensionality-reduction/03-t-sne/) |
| 06-04 | [UMAP](../06-dimensionality-reduction/04-umap/) |
| 06-05 | [Sélection de variables : filtres, wrappers, méthodes embarquées](../06-dimensionality-reduction/05-feature-selection/) |

## Partie 7 : Réseaux de neurones

| # | Notebook |
|---|---|
| 07-01 | [Le perceptron](../07-neural-networks/01-the-perceptron/) |
| 07-02 | [Perceptron multicouche et rétropropagation, en NumPy](../07-neural-networks/02-mlp-and-backpropagation/) |
| 07-03 | [Le même réseau en PyTorch](../07-neural-networks/03-the-same-net-in-pytorch/) |
| 07-04 | [Les fonctions d'activation et pourquoi elles comptent](../07-neural-networks/04-activation-functions/) |
| 07-05 | [Optimiseurs : SGD, Momentum, RMSProp, Adam](../07-neural-networks/05-optimisers/) |
| 07-06 | [Régularisation : dropout, batch norm, weight decay, early stopping](../07-neural-networks/06-regularisation/) |
| 07-07 | [Une boucle d'entraînement réutilisable](../07-neural-networks/07-a-training-loop/) |

## Partie 8 : Vision par ordinateur

| # | Notebook |
|---|---|
| 08-01 | [Convolution, pooling, et ce qu'apprend un filtre](../08-computer-vision/01-convolution-and-pooling/) |
| 08-02 | [Un CNN sur Fashion-MNIST, couche par couche](../08-computer-vision/02-a-cnn-layer-by-layer/) |
| 08-03 | [Architectures classiques : LeNet, VGG, ResNet](../08-computer-vision/03-classic-architectures/) |
| 08-04 | [Augmentation de données](../08-computer-vision/04-data-augmentation/) |
| 08-05 | [Transfer learning et fine-tuning](../08-computer-vision/05-transfer-learning/) |
| 08-06 | [La segmentation d'images, une introduction](../08-computer-vision/06-image-segmentation/) |
| 08-07 | [La détection d'objets, une introduction](../08-computer-vision/07-object-detection/) |

## Partie 9 : Séquences et langage

| # | Notebook |
|---|---|
| 09-01 | [Prétraitement du texte, sac de mots et TF-IDF](../09-sequences-and-language/01-text-preprocessing-and-tfidf/) |
| 09-02 | [Word embeddings : Word2Vec et GloVe](../09-sequences-and-language/02-word-embeddings/) |
| 09-03 | [Réseaux de neurones récurrents](../09-sequences-and-language/03-recurrent-neural-networks/) |
| 09-04 | [LSTM](../09-sequences-and-language/04-lstm/) |
| 09-05 | [GRU, et ce qui le distingue du LSTM](../09-sequences-and-language/05-gru/) |
| 09-06 | [Séquence à séquence avec attention](../09-sequences-and-language/06-seq2seq-with-attention/) |
| 09-07 | [Le Transformer, construit de zéro](../09-sequences-and-language/07-the-transformer/) |
| 09-08 | [Fine-tuning d'un transformer préentraîné](../09-sequences-and-language/08-fine-tuning/) |
| 09-09 | [Prévision de séries temporelles : ARIMA contre ML contre deep learning](../09-sequences-and-language/09-time-series-forecasting/) |

## Partie 10 : Modèles génératifs

| # | Notebook |
|---|---|
| 10-01 | [Auto-encodeurs](../10-generative-models/01-autoencoders/) |
| 10-02 | [Auto-encodeurs variationnels](../10-generative-models/02-variational-autoencoders/) |
| 10-03 | [Réseaux antagonistes génératifs](../10-generative-models/03-generative-adversarial-networks/) |
| 10-04 | [Modèles de diffusion, le plus petit exemple qui marche](../10-generative-models/04-diffusion-models/) |

## Partie 11 : Apprentissage par renforcement

| # | Notebook |
|---|---|
| 11-01 | [Le cadre de l'apprentissage par renforcement : agents, états, récompenses](../11-reinforcement-learning/01-the-setup/) |
| 11-02 | [Bandits à plusieurs bras](../11-reinforcement-learning/02-multi-armed-bandits/) |
| 11-03 | [Processus de décision markoviens, itération sur la valeur et sur la politique](../11-reinforcement-learning/03-markov-decision-processes/) |
| 11-04 | [Q-learning](../11-reinforcement-learning/04-q-learning/) |
| 11-05 | [SARSA, et en quoi il diffère de Q-learning](../11-reinforcement-learning/05-sarsa/) |
| 11-06 | [Deep Q-Networks](../11-reinforcement-learning/06-deep-q-networks/) |
| 11-07 | [Gradients de politique et REINFORCE](../11-reinforcement-learning/07-policy-gradients/) |
| 11-08 | [Acteur-critique et PPO](../11-reinforcement-learning/08-actor-critic-and-ppo/) |

## Partie 12 : Tout mettre ensemble

| # | Notebook |
|---|---|
| 12-01 | [**Le tableau des résultats** : chaque méthode sur chaque jeu de données maison](../12-putting-it-together/01-the-scoreboard/) |
| 12-02 | [Interpréter les modèles : importance par permutation, SHAP, LIME](../12-putting-it-together/02-interpreting-models/) |
| 12-03 | [Les pipelines, et en finir avec les fuites de données](../12-putting-it-together/03-pipelines/) |
| 12-04 | [Sauvegarder, charger et servir un modèle](../12-putting-it-together/04-saving-and-serving/) |
| 12-05 | [Les erreurs que tout le monde fait](../12-putting-it-together/05-common-mistakes/) |

---

## Les jeux de données maison

Tous les notebooks utilisent le même petit ensemble, pour qu'une comparaison d'un
chapitre à l'autre veuille dire quelque chose. Vous voyez tout de suite en quoi
un SVM et une forêt aléatoire diffèrent, puisqu'on leur a posé la même question.

| Jeu de données | Tâche | Taille | Pourquoi celui-là |
|---|---|---|---|
| California Housing | régression | 20,640 × 8 | Fourni avec scikit-learn, donc le chapitre un tourne sans rien télécharger |
| Breast Cancer Wisconsin | classification binaire | 569 × 30 | Assez petit pour que chaque méthode s'entraîne instantanément |
| **UCI Dry Bean** | classification à 7 classes | 13,611 × 16 | Publié en 2020 et presque absent des tutoriels, ce qui sort le livre des sentiers battus |
| UCI Bike Sharing | régression, séries temporelles | 17,379 × 16 | Un seul jeu de données qui sert à la fois à la régression tabulaire et aux modèles de séquences |
| Fashion-MNIST | classification d'images | 70,000 × 28 × 28 | Même forme que MNIST, plusieurs fois plus difficile, donc les résultats des CNN ne sont pas tous à 99 % |

Sources, licences et dates de récupération sont dans
[`data/README.md`](../data/README.md).

## Le tableau des résultats

La partie 12 rassemble dans un seul tableau le résultat de chaque méthode sur
chaque jeu de données maison, puis explique la logique. Une partie est
prévisible, une autre pas : le gradient boosting mène généralement sur données
tabulaires, les k plus proches voisins s'effondrent quand les colonnes se
multiplient, un modèle linéaire bat un réseau de neurones quand les lignes sont
peu nombreuses. Tout l'intérêt du livre est que vous en sortiez en sachant
*pourquoi*, et pas seulement *lequel*.

---

Par **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

---

## Réutiliser, et aider le livre à circuler

Le code est sous licence MIT. Copiez-le, adaptez-le, mettez-le dans vos propres
projets ou dans votre enseignement, sans demander la permission et sans
obligation de citer la source. Le texte et les figures sont en CC BY 4.0, donc
ils circulent tout aussi librement, en citant l'auteur. Si un chapitre vous
fait gagner un après-midi, c'est qu'il a fait son travail.

Si vous le trouvez utile, **une étoile aide les autres à le trouver**, et c'est à
peu près le seul moyen pour qu'un livre comme celui-ci circule. Les corrections
et les désaccords sont les bienvenus dans les issues, surtout si vous relancez
un calcul et obtenez un autre résultat.

---
