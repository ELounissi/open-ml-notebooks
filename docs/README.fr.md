[English](../README.md) · [العربية](README.ar.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Português](README.pt.md) · [हिन्दी](README.hi.md) · [简体中文](README.zh-CN.md)

# open-ml-notebooks

### Apprendre le machine learning en lisant le code

Un livre libre et gratuit de **86 notebooks Jupyter** sur le machine learning, le
deep learning et l'apprentissage par renforcement : un notebook, une méthode,
expliquée simplement, codée de zéro, puis reprise proprement avec la bibliothèque.

Chaque notebook est livré **déjà exécuté**. Ouvrez-en un sur GitHub : les
graphiques, les chiffres et les sorties sont là. Aucune installation pour
commencer à lire.

**Par [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)**

Les notebooks eux-mêmes sont en anglais, et ce n'est pas vraiment un obstacle :
le code, les noms de variables, les tableaux affichés et les légendes des
graphiques sont en anglais de toute façon, puisque c'est la langue de toutes les
bibliothèques de ML et de tous les messages d'erreur. Les courbes et les chiffres,
eux, se lisent pareil dans n'importe quelle langue. Cette page et le programme
sont traduits pour que vous sachiez où aller ; le reste se suit très bien sans
lire l'anglais couramment.

![Une droite ne voit pas les deux heures de pointe](../02-regression/01-linear-regression/figures/fig-04-where-it-loses.png)

*Extrait de [Régression linéaire](../02-regression/01-linear-regression/) :
pourquoi le modèle le plus simple échoue sur des données cycliques, et la
correction d'une seule ligne qui le fait passer de 0,39 à 0,68.*

---

## Pourquoi ce livre

La plupart des tutoriels vous montrent quelle fonction appeler. Peu expliquent ce
que la méthode fait réellement, et presque aucun ne dit **quand c'est le mauvais
choix**.

Chaque notebook répond ici à quatre questions :

1. **Que fait vraiment cette méthode ?** Avec des mots d'abord, et une image.
2. **Quelles sont les maths ?** Écrites en entier, et seulement ce qui sert.
3. **Puis-je la coder moi-même ?** Une version minimale en NumPy, pour que rien
   ne reste magique.
4. **Quand gagne-t-elle, et quand perd-elle ?** Mesuré, pas affirmé.

La quatrième est celle qui compte, et c'est celle qui manque presque toujours.

## Le fil conducteur

Tous les notebooks utilisent les **mêmes cinq jeux de données**. C'est voulu.
Quand un SVM obtient 0,93 au chapitre trois et une forêt aléatoire 0,94 au
chapitre quatre, les deux chiffres se comparent sans réserve, parce que la
question posée était exactement la même.

| Jeu de données | Tâche | Taille | Pourquoi celui-là |
|---|---|---|---|
| California Housing | régression | 20,640 × 8 | Fourni avec scikit-learn : le chapitre un tourne sans rien télécharger |
| Breast Cancer Wisconsin | classification binaire | 569 × 30 | Assez petit pour que chaque méthode s'entraîne instantanément |
| **UCI Dry Bean** | classification à 7 classes | 13,611 × 16 | Publié en 2020, presque absent des tutoriels, et agréablement déséquilibré |
| UCI Bike Sharing | régression + séries temporelles | 17,379 × 16 | Un seul jeu de données pour les chapitres tabulaires et les chapitres séquentiels |
| Fashion-MNIST | classification d'images | 70,000 images | La forme de MNIST, plusieurs fois plus difficile, donc les CNN restent intéressants |

À la fin, [**le tableau des résultats**](CURRICULUM.fr.md) réunit chaque méthode
face à chaque jeu de données dans un seul tableau et explique la logique :
pourquoi le gradient boosting l'emporte le plus souvent sur données tabulaires,
pourquoi les k plus proches voisins s'effondrent quand les colonnes se
multiplient, pourquoi un modèle linéaire bat un réseau de neurones quand les
lignes sont peu nombreuses.

---

## Sommaire

Programme complet avec les 86 notebooks : **[CURRICULUM.fr.md](CURRICULUM.fr.md)**

| Partie | Contenu |
|---|---|
| **1 : [Fondamentaux](../01-foundations/)** | La démarche de bout en bout, train/validation/test, surapprentissage, validation croisée, métriques, mise à l'échelle, encodage, données manquantes, réglage des hyperparamètres |
| **2 : [Régression](../02-regression/)** | Linéaire, descente de gradient, polynomiale, Ridge, Lasso, Elastic Net, Huber, RANSAC, quantile, GLM |
| **3 : [Classification](../03-classification/)** | Régression logistique, k-NN, Naive Bayes, LDA/QDA, SVM et noyaux, arbres de décision, déséquilibre, calibration |
| **4 : [Ensembles](../04-ensembles/)** | Bagging, forêts aléatoires, Extra Trees, AdaBoost, gradient boosting, XGBoost, LightGBM, CatBoost, stacking |
| **5 : [Non supervisé](../05-unsupervised/)** | k-Means, choix de k, clustering hiérarchique, DBSCAN, HDBSCAN, mélanges gaussiens, détection d'anomalies, Apriori |
| **6 : [Réduction de dimension](../06-dimensionality-reduction/)** | ACP (PCA), Kernel PCA, ICA, NMF, t-SNE, UMAP, sélection de variables |
| **7 : [Réseaux de neurones](../07-neural-networks/)** | Perceptron, MLP et rétropropagation en NumPy, PyTorch, activations, optimiseurs, régularisation |
| **8 : [Vision par ordinateur](../08-computer-vision/)** | Convolution, CNN, LeNet/VGG/ResNet, augmentation, transfer learning, segmentation, détection |
| **9 : [Séquences et langage](../09-sequences-and-language/)** | TF-IDF, embeddings, RNN, LSTM, GRU, seq2seq, attention, Transformers, séries temporelles |
| **10 : [Modèles génératifs](../10-generative-models/)** | Auto-encodeurs, VAE, GAN, diffusion |
| **11 : [Apprentissage par renforcement](../11-reinforcement-learning/)** | Bandits, MDP, Q-learning, SARSA, DQN, gradients de politique, acteur-critique, PPO |
| **12 : [Tout mettre ensemble](../12-putting-it-together/)** | Le tableau des résultats, SHAP et LIME, pipelines, mise en production, erreurs classiques |

### Quelques-uns à ouvrir en premier

- **[01-03 : Surapprentissage et sous-apprentissage](../01-foundations/03-overfitting-and-underfitting/)**,
  le diagnostic le plus utile de tout le machine learning, dessiné plutôt que
  défini, avec au passage l'ajustement de degré 20 qui fait moins bien que le
  degré 4 sur l'ensemble d'entraînement comme sur l'ensemble de test, parce que la
  matrice de design a un conditionnement de `1.1e+21` et que le solveur abandonne
  discrètement avant même que le surapprentissage ait sa chance
- **[02-01 : Régression linéaire](../02-regression/01-linear-regression/)**,
  l'équation normale démontrée, une version NumPy de 30 lignes qui colle à
  scikit-learn jusqu'à la douzième décimale, des diagnostics de résidus qui
  révèlent un biais que la statistique résumée cachait, et la méthode qui échoue
  sur des données cycliques avant d'être sauvée par l'encodage plutôt que par un
  modèle plus gros
- **[02-04 : Régression Ridge](../02-regression/04-ridge-regression/)**, la
  pénalité L2, et un constat honnête : elle n'a **rien apporté en accuracy** sur
  ces données, tout en rendant les coefficients **32× plus stables** ; plus une
  expérience sur la mise à l'échelle qui est sortie à l'envers et qui s'explique
  toute seule
- **[02-05 : Régression Lasso](../02-regression/05-lasso-regression/)**, pourquoi
  un seul exposant transforme un rétrécissement en sélection, et la découverte que
  **`LassoCV` a gardé 17 colonnes de bruit pur sur 30**, parce que la validation
  croisée optimise la prédiction et pas la parcimonie
- **[03-01 : Régression logistique](../03-classification/01-logistic-regression/)**,
  la sigmoïde, la log loss, la descente de gradient écrite en entier, un softmax
  sur sept variétés de haricots déséquilibrées, et pourquoi c'est la classe la
  *plus rare* qui s'est révélée la plus facile
- **[03-02 : k plus proches voisins](../03-classification/02-k-nearest-neighbours/)**,
  le modèle qui ne s'entraîne pas du tout ; la mise à l'échelle vaut à elle seule
  **+0,205 d'accuracy**, `k=1` obtient un parfait 1,000 sur les données
  d'entraînement et ce chiffre ne veut rien dire, et le fléau de la dimension est
  mesuré au lieu d'être asséné
- **[03-03 : Naive Bayes](../03-classification/03-naive-bayes/)**, pourquoi une
  hypothèse que tout le monde sait fausse fonctionne quand même, et un détour en
  treize points sur la valeur par défaut de `var_smoothing` dans scikit-learn, qui
  détruit sans bruit les variables de petite échelle
- **[03-05 : Machines à vecteurs de support](../03-classification/05-support-vector-machines/)**,
  la marge, ce que `C` contrôle réellement, et l'astuce du noyau montrée en
  soulevant deux anneaux dans une troisième dimension, avant d'expliquer pourquoi
  on n'a jamais besoin de la construire ; plus le coût dont personne ne parle,
  où 27× plus de lignes coûtent **139× plus de temps**
- **[03-06 : Arbres de décision](../03-classification/06-decision-trees/)**, la
  recherche de coupure écrite de zéro, la meilleure question de tout le jeu de
  données trouvée par le calcul, un arbre de profondeur 3 qui se lit à l'œil, et
  un arbre sans contrainte qui atteint la perfection sur l'entraînement tout en
  devenant *moins bon* sur les données mises de côté
- **[04-02 : Forêts aléatoires](../04-ensembles/02-random-forest/)**, l'équation
  de la variance qui explique toute la conception, le score out-of-bag, deux
  mesures d'importance des variables qui ne sont pas d'accord, et un duel contre
  la régression logistique que la forêt **ne gagne pas**
- **[04-05 : Gradient boosting](../04-ensembles/05-gradient-boosting/)**, pourquoi
  ajuster le résidu *est* une descente de gradient, l'arbitrage sur le taux
  d'apprentissage, et la différence la plus nette avec une forêt : passé un
  certain point, ajouter des arbres rend le boosting **moins bon**
- **[05-01 : k-Means](../05-unsupervised/01-k-means/)**, l'algorithme de Lloyd
  codé de zéro, le coude et la silhouette qui désignent tous les deux le *mauvais*
  nombre de groupes sur des données dont on connaît la vérité, et quatre façons
  d'échouer, dont k-means qui découpe du bruit pur en groupes bien nets sans la
  moindre hésitation
- **[05-04 : DBSCAN et HDBSCAN](../05-unsupervised/04-dbscan-and-hdbscan/)**, le
  clustering par densité qui trouve des croissants hors de portée de k-means, le
  choix d'`eps` à partir du coude de la courbe des k-distances plutôt qu'au jugé,
  et la découverte qu'**aucun réglage d'`eps` ne dit "ces données n'ont pas de
  structure"**
- **[06-01 : ACP (PCA)](../06-dimensionality-reduction/01-principal-component-analysis/)**,
  la décomposition en valeurs propres faite à la main, qui rejoint scikit-learn à
  2,2×10⁻¹⁶ près, des composantes qui finissent par vouloir dire "taille" et
  "forme", et un constat honnête : l'ACP n'a **jamais fait mieux** que garder
  simplement toutes les colonnes
- **[07-01 : Le perceptron](../07-neural-networks/01-the-perceptron/)**, un seul
  neurone, la règle d'apprentissage qui ne demande aucun calcul différentiel, et
  les quatre points du XOR qui ont bloqué les réseaux de neurones pendant dix ans
- **[07-02 : MLP et rétropropagation](../07-neural-networks/02-mlp-and-backpropagation/)**,
  chaque gradient dérivé à la main puis **vérifié numériquement à 1,9×10⁻¹⁰
  près**, et un réseau qui fonctionne, en NumPy seul
- **[07-03 : Le même réseau en PyTorch](../07-neural-networks/03-the-same-net-in-pytorch/)**,
  autograd démontré sur un cas vérifiable à la main, la boucle d'entraînement en
  cinq lignes que l'on réutilise toute sa vie, et les trois erreurs que tout le
  monde fait une fois
- **[08-02 : Un CNN, couche par couche](../08-computer-vision/02-a-cnn-layer-by-layer/)**,
  le partage de poids expliqué, et un réseau convolutif qui bat un réseau dense
  en accuracy, en nombre de paramètres **et** en vitesse ; ses filtres de première
  couche deviennent des détecteurs de contours que personne ne lui a demandés
- **[11-04 : Q-learning](../11-reinforcement-learning/04-q-learning/)**, Cliff
  Walking construit de zéro sans dépendre de `gym`, la mise à jour de Bellman en
  une ligne, et le résultat classique où Q-learning trouve la meilleure politique
  pendant que SARSA récolte plus de récompense

Ensemble, ils couvrent la régression et la classification supervisées, les
ensembles, le clustering non supervisé, la réduction de dimension et
l'apprentissage par renforcement, et chacun d'eux est terminé et exécuté.

### Le tableau des résultats

Comme tous les notebooks partagent les mêmes jeux de données, les comparaisons
s'accumulent. Sur **UCI Dry Bean**, accuracy en validation croisée à 5 plis :

| Méthode | Accuracy |
|---|---|
| SVM, noyau RBF | **0.9301** |
| Gradient boosting | 0.9271 |
| SVM, noyau linéaire | 0.9262 |
| Forêt aléatoire | 0.9244 |
| Régression logistique | 0.9234 |
| k plus proches voisins | 0.9231 |
| Réseau de neurones (NumPy, de zéro) | 0.9306 * |
| Naive Bayes | 0.8972 |
| Arbre de décision | 0.8945 |

\* un seul découpage avec ensemble de test mis de côté, et non 5 plis : la
comparaison ne tient donc pas. Voir le
[notebook](../07-neural-networks/02-mlp-and-backpropagation/) pour comprendre
pourquoi je n'en fais pas une victoire.

Neuf méthodes tiennent dans quatre centièmes, et une simple droite en bat la
plupart. Les mesures des haricots sont une géométrie lisse et corrélée : les
modèles souples n'ont presque aucune structure non linéaire à exploiter.

Changez de jeu de données et le classement change. Sur **California Housing**, le
gradient boosting fait passer le RMSE de 0,7263 pour la régression linéaire à
**0,4668**, soit 36 % de mieux, parce que ce problème-là *est* plein
d'interactions.

**La méthode qui gagne est une propriété de vos données, pas un classement
d'algorithmes.** C'est le fil qui traverse tout le livre, et plusieurs notebooks
rapportent ici un résultat que j'attendais dans l'autre sens, et le disent sans
détour.

---

## Pour commencer

```bash
git clone https://github.com/ELounissi/open-ml-notebooks
cd open-ml-notebooks
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Tout tourne **en quelques minutes sur le CPU d'un portable**. Un GPU accélère les
chapitres de deep learning, mais n'est jamais obligatoire. Les petits jeux de
données sont versionnés avec le dépôt, donc la plupart des notebooks tournent
hors ligne immédiatement.

**Grand débutant ?** Commencez par la [partie 1, Fondamentaux](../01-foundations/).
**Vous connaissez les bases ?** Allez directement à la méthode qu'il vous faut :
chaque notebook se lit seul.
**Vous préparez des entretiens ?** La section « quand ça gagne, quand ça perd »
de chaque notebook est écrite exactement pour cette discussion.

---

## Réutiliser tout ça

Le code est sous licence **MIT** : copiez-le, adaptez-le, mettez-le dans vos
projets ou dans vos cours, sans demander la permission et sans obligation de
citer la source. Le texte et les figures sont en **CC BY 4.0** : reprenez-les
dans un cours, une présentation ou un groupe de travail, en citant l'auteur. Les
jeux de données gardent leurs propres licences, notées un par un dans
[`data/README.md`](../data/README.md).

Si ça vous a fait gagner du temps, **une étoile aide les autres à le trouver**.
Une erreur ? Ouvrez une issue : les corrections sont vraiment les bienvenues,
surtout celles qui montrent qu'une affirmation d'ici est fausse.

---

### Thèmes

`machine-learning` `deep-learning` `reinforcement-learning` `machine-learning-tutorial`
`jupyter-notebook` `python` `scikit-learn` `pytorch` `data-science` `neural-network`
`cnn` `rnn` `lstm` `transformer` `linear-regression` `logistic-regression`
`random-forest` `xgboost` `clustering` `pca` `learn-machine-learning`
`machine-learning-algorithms` `ml-tutorial` `data-science-tutorial` `education`

`#MachineLearning` `#DeepLearning` `#ReinforcementLearning` `#DataScience`
`#Python` `#ScikitLearn` `#PyTorch` `#LearnMachineLearning` `#MLTutorial`
`#JupyterNotebook` `#OpenSource` `#AI`

---

**Par Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)
