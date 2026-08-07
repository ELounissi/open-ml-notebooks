[English](../README.md) · [العربية](README.ar.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Português](README.pt.md) · [हिन्दी](README.hi.md) · [简体中文](README.zh-CN.md)

# open-ml-notebooks

### Machine Learning lernen, indem du den Code liest

Ein kostenloses, offenes Buch aus **86 Jupyter-Notebooks** über Machine Learning,
Deep Learning und Reinforcement Learning: ein Verfahren pro Notebook, in
einfacher Sprache erklärt, zuerst von Hand implementiert und dann ordentlich mit
der Bibliothek gemacht.

Jedes Notebook ist **bereits ausgeführt**. Öffne eines auf GitHub, und die
Diagramme, Zahlen und Ausgaben stehen direkt da. Zum Lesen musst du nichts
installieren.

**Von [Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)**

Die Notebooks selbst sind auf Englisch, und das steht dir kaum im Weg. Code,
Variablennamen, die ausgegebenen Tabellen und die Achsenbeschriftungen sind
ohnehin in der Sprache, die jede ML-Bibliothek und jede Fehlermeldung spricht,
und Diagramme und Zahlen liest man in jeder Sprache gleich. Du kannst also jedem
Ergebnis folgen, ohne englischen Fließtext sicher zu beherrschen.

![Eine Gerade kann zwei Pendlerspitzen nicht sehen](../02-regression/01-linear-regression/figures/fig-04-where-it-loses.png)

*Aus [Lineare Regression](../02-regression/01-linear-regression/): warum das
einfachste Modell an zyklischen Daten scheitert, und die Korrektur in einer
Zeile, die es von 0,39 auf 0,68 bringt.*

---

## Warum dieses Buch

Die meisten Tutorials zeigen dir, welche Funktion du aufrufen sollst. Wenige
zeigen, was das Verfahren dabei tatsächlich tut, und fast keines sagt dir, **wann
es die falsche Wahl ist**.

Jedes Notebook hier beantwortet vier Fragen:

1. **Was macht dieses Verfahren eigentlich?** Erst in Worten, mit einem Bild.
2. **Wie sieht die Mathematik aus?** Ausgeschrieben, nur so viel wie nötig.
3. **Kann ich es selbst bauen?** Eine minimale Fassung in NumPy, damit nichts nach Magie aussieht.
4. **Wann gewinnt es, und wann verliert es?** Gemessen, nicht behauptet.

Die vierte Frage ist die, auf die es ankommt, und die, die üblicherweise fehlt.

## Der rote Faden

Jedes Notebook benutzt **dieselben fünf Datensätze**. Das ist Absicht. Wenn eine
Support Vector Machine in Kapitel drei 0,93 erreicht und ein Random Forest in
Kapitel vier 0,94, dann sind die Zahlen direkt vergleichbar, weil die Frage
identisch war.

| Datensatz | Aufgabe | Größe | Warum dieser |
|---|---|---|---|
| California Housing | Regression | 20.640 × 8 | Liegt scikit-learn bei: Kapitel eins läuft ohne Download |
| Breast Cancer Wisconsin | binäre Klassifikation | 569 × 30 | Klein genug, dass jedes Verfahren sofort trainiert |
| **UCI Dry Bean** | Klassifikation mit 7 Klassen | 13.611 × 16 | 2020 veröffentlicht, in Tutorials kaum benutzt und angenehm unausgewogen |
| UCI Bike Sharing | Regression und Zeitreihe | 17.379 × 16 | Ein Datensatz für die Tabellenkapitel und die Sequenzkapitel zugleich |
| Fashion-MNIST | Bildklassifikation | 70.000 Bilder | Form von MNIST, um einiges schwerer, damit CNNs interessant bleiben |

Am Ende stellt [**die Ergebnistabelle**](CURRICULUM.de.md) jedes Verfahren jedem
Datensatz gegenüber und erklärt das Muster: warum Gradient Boosting bei
Tabellendaten meistens vorn liegt, warum k-nächste Nachbarn zerfallen, sobald die
Spalten mehr werden, und warum ein lineares Modell ein neuronales Netz schlägt,
wenn es wenige Zeilen gibt.

---

## Inhalt

Vollständiges Inhaltsverzeichnis mit allen 86 Notebooks: **[CURRICULUM.de.md](CURRICULUM.de.md)**

| Teil | Inhalt |
|---|---|
| **1: [Grundlagen](../01-foundations/)** | Der Ablauf, Training/Validierung/Test, Overfitting, Kreuzvalidierung, Metriken, Skalierung, Kodierung, fehlende Werte, Tuning |
| **2: [Regression](../02-regression/)** | Linear, Gradientenabstieg, polynomial, Ridge, Lasso, Elastic Net, Huber, RANSAC, Quantile, GLMs |
| **3: [Klassifikation](../03-classification/)** | Logistische Regression, k-NN, Naive Bayes, LDA/QDA, SVM und Kernel, Entscheidungsbäume, Ungleichgewicht, Kalibrierung |
| **4: [Ensembles](../04-ensembles/)** | Bagging, Random Forest, Extra Trees, AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost, Stacking |
| **5: [Unüberwachtes Lernen](../05-unsupervised/)** | k-Means, k wählen, hierarchisch, DBSCAN, HDBSCAN, Gaußsche Mischverteilungen, Anomalieerkennung, Apriori |
| **6: [Dimensionsreduktion](../06-dimensionality-reduction/)** | PCA, Kernel-PCA, ICA, NMF, t-SNE, UMAP, Feature-Auswahl |
| **7: [Neuronale Netze](../07-neural-networks/)** | Perzeptron, MLP und Backpropagation in NumPy, PyTorch, Aktivierungen, Optimierer, Regularisierung |
| **8: [Computer Vision](../08-computer-vision/)** | Faltung, CNNs, LeNet/VGG/ResNet, Augmentierung, Transfer Learning, Segmentierung, Detektion |
| **9: [Sequenzen und Sprache](../09-sequences-and-language/)** | TF-IDF, Embeddings, RNN, LSTM, GRU, seq2seq, Attention, Transformer, Zeitreihen |
| **10: [Generative Modelle](../10-generative-models/)** | Autoencoder, VAEs, GANs, Diffusion |
| **11: [Reinforcement Learning](../11-reinforcement-learning/)** | Bandits, MDPs, Q-Learning, SARSA, DQN, Policy Gradients, Actor-Critic, PPO |
| **12: [Alles zusammen](../12-putting-it-together/)** | Die Ergebnistabelle, SHAP und LIME, Pipelines, Deployment, häufige Fehler |

### Ein paar, die sich zuerst lohnen

- **[01-03: Overfitting und Underfitting](../01-foundations/03-overfitting-and-underfitting/)**,
  die nützlichste Diagnose im Machine Learning, gezeichnet statt definiert,
  samt der Anpassung mit Grad 20, die sowohl auf den Trainingsdaten als auch auf
  den Testdaten schlechter ist als Grad 4, weil die Designmatrix eine
  Konditionszahl von `1.1e+21` hat und der Solver still aufgibt, bevor Overfitting
  überhaupt eine Chance bekommt
- **[02-01: Lineare Regression](../02-regression/01-linear-regression/)**, die
  hergeleitete Normalengleichung, eine NumPy-Fassung in 30 Zeilen, die auf 12
  Nachkommastellen mit scikit-learn übereinstimmt, Residuendiagnostik, die eine
  Verzerrung offenlegt, welche die Kennzahl verdeckt hatte, und das Verfahren, das
  an zyklischen Daten scheitert und dann durch Kodierung statt durch ein größeres
  Modell gerettet wird
- **[02-04: Ridge-Regression](../02-regression/04-ridge-regression/)**, die
  L2-Strafe und der ehrliche Befund, dass sie auf diesen Daten **überhaupt keine
  Genauigkeit** gebracht hat, die Koeffizienten aber **32× stabiler** macht; dazu
  ein Skalierungsexperiment, das andersherum ausging als erwartet und sich selbst
  erklärt
- **[02-05: Lasso-Regression](../02-regression/05-lasso-regression/)**, warum ein
  einziger Exponent aus Schrumpfung eine Auswahl macht, und die Entdeckung, dass
  **`LassoCV` 17 von 30 reinen Rauschspalten behalten hat**, weil Kreuzvalidierung
  auf Vorhersage optimiert und nicht auf Sparsity
- **[03-01: Logistische Regression](../03-classification/01-logistic-regression/)**,
  die Sigmoidfunktion, Log Loss, ausgeschriebener Gradientenabstieg, Softmax auf
  sieben unterschiedlich häufigen Bohnensorten, und warum ausgerechnet die
  *seltenste* Klasse die einfachste war
- **[03-02: k-nächste Nachbarn](../03-classification/02-k-nearest-neighbours/)**,
  das Modell, das gar nicht trainiert; allein die Skalierung ist **+0,205
  Genauigkeit** wert, `k=1` erreicht auf den Trainingsdaten eine bedeutungslose
  perfekte 1,000, und der Fluch der Dimensionalität wird gemessen statt behauptet
- **[03-03: Naive Bayes](../03-classification/03-naive-bayes/)**, warum eine
  Annahme, von der jeder weiß, dass sie falsch ist, trotzdem funktioniert, und ein
  Abstecher in dreizehn Punkten zur Voreinstellung `var_smoothing` in
  scikit-learn, die kleinskalige Features klammheimlich zerstört
- **[03-05: Support Vector Machines](../03-classification/05-support-vector-machines/)**,
  die Margin, was `C` wirklich steuert, und der Kernel-Trick, gezeigt an zwei
  Ringen, die in eine dritte Dimension gehoben werden, bevor erklärt wird, warum
  man sie nie wirklich bauen muss; dazu die Kosten, die niemand erwähnt: 27× so
  viele Zeilen kosten **139× so viel Zeit**
- **[03-06: Entscheidungsbäume](../03-classification/06-decision-trees/)**, die
  Suche nach dem Split von Grund auf geschrieben, die beste einzelne Frage im
  ganzen Datensatz durch bloßes Rechnen gefunden, ein lesbarer Baum der Tiefe 3 und ein
  ungebremster, der perfekte Trainingsgenauigkeit erreicht und auf
  zurückgehaltenen Daten dabei *schlechter* wird
- **[04-02: Random Forests](../04-ensembles/02-random-forest/)**, die
  Varianzgleichung, die das ganze Design erklärt, Out-of-Bag-Bewertung, zwei Maße
  für Feature-Importance, die sich widersprechen, und ein direkter Vergleich mit
  logistischer Regression, den der Forest **nicht gewinnt**
- **[04-05: Gradient Boosting](../04-ensembles/05-gradient-boosting/)**, warum das
  Anpassen an das Residuum *genau* Gradientenabstieg ist, was die Lernrate kostet
  und was sie bringt, und der schärfste Unterschied zum Forest: irgendwann machen
  weitere Bäume das Boosting **schlechter**
- **[05-01: k-Means](../05-unsupervised/01-k-means/)**, Lloyds Algorithmus von
  Grund auf, Ellenbogen und Silhouette, die beide auf die *falsche* Clusterzahl
  zeigen, obwohl die Wahrheit bekannt ist, und vier Arten, auf die es schiefgeht,
  darunter k-Means, das reines Rauschen selbstbewusst in ordentliche Gruppen zerlegt
- **[05-04: DBSCAN und HDBSCAN](../05-unsupervised/04-dbscan-and-hdbscan/)**,
  dichtebasiertes Clustering, das Sicheln findet, an denen k-Means scheitert,
  `eps` aus dem Ellenbogen der k-Distanz gewählt statt geraten, und die
  Entdeckung, dass **keine `eps`-Einstellung meldet: "diese Daten haben keine
  Struktur"**
- **[06-01: PCA](../06-dimensionality-reduction/01-principal-component-analysis/)**,
  Eigenzerlegung von Hand, die mit scikit-learn auf 2,2×10⁻¹⁶ übereinstimmt,
  Komponenten, die sich als "Größe" und "Form" entpuppen, und der ehrliche Befund,
  dass PCA es **nie geschlagen** hat, einfach alle Spalten zu behalten
- **[07-01: Das Perzeptron](../07-neural-networks/01-the-perceptron/)**, ein
  Neuron, die Lernregel, die ohne Analysis auskommt, und die vier Punkte von XOR,
  die neuronale Netze ein Jahrzehnt lang aufgehalten haben
- **[07-02: MLP und Backpropagation](../07-neural-networks/02-mlp-and-backpropagation/)**,
  jeder Gradient von Hand hergeleitet und **numerisch auf 1,9×10⁻¹⁰ geprüft**,
  danach ein funktionierendes Netz allein in NumPy
- **[07-03: Dasselbe Netz in PyTorch](../07-neural-networks/03-the-same-net-in-pytorch/)**,
  Autograd vorgeführt an etwas, das man von Hand nachrechnen kann, die
  Trainingsschleife aus fünf Zeilen, die du für immer wiederverwendest, und die
  drei Fehler, die jeder einmal macht
- **[08-02: Ein CNN, Schicht für Schicht](../08-computer-vision/02-a-cnn-layer-by-layer/)**,
  Weight Sharing erklärt, und ein Faltungsnetz, das ein dichtes Netz bei
  Genauigkeit, Parameterzahl **und** Geschwindigkeit schlägt; seine Filter der
  ersten Schicht werden zu Kantendetektoren, um die niemand gebeten hat
- **[11-04: Q-Learning](../11-reinforcement-learning/04-q-learning/)**, Cliff
  Walking von Grund auf gebaut, ohne Abhängigkeit von `gym`, das Bellman-Update in
  einer Zeile, und das klassische Ergebnis, bei dem Q-Learning die bessere
  Strategie findet, während SARSA mehr Belohnung einsammelt

Zusammen decken sie überwachte Regression, überwachte Klassifikation, Ensembles,
unüberwachtes Clustering, Dimensionsreduktion und Reinforcement Learning ab, und
jedes einzelne davon ist fertig und ausgeführt.

### Die Ergebnistabelle

Weil jedes Notebook dieselben Datensätze benutzt, summieren sich die Vergleiche.
Auf **UCI Dry Bean**, Genauigkeit aus 5-facher Kreuzvalidierung:

| Verfahren | Genauigkeit |
|---|---|
| SVM, RBF-Kernel | **0.9301** |
| Gradient Boosting | 0.9271 |
| SVM, linearer Kernel | 0.9262 |
| Random Forest | 0.9244 |
| Logistische Regression | 0.9234 |
| k-nächste Nachbarn | 0.9231 |
| Neuronales Netz (NumPy, von Hand) | 0.9306 * |
| Naive Bayes | 0.8972 |
| Entscheidungsbaum | 0.8945 |

\* eine einzelne zurückgehaltene Aufteilung statt 5-facher Kreuzvalidierung, also
nicht direkt vergleichbar. Im [Notebook](../07-neural-networks/02-mlp-and-backpropagation/)
steht, warum ich das nicht als Sieg verbuche.

Die Zahlen in dieser Tabelle stehen genau so da, wie die Notebooks sie ausgeben,
also mit Punkt als Dezimaltrennzeichen; im Fließtext dieser Seite steht das
übliche deutsche Komma.

Neun Verfahren liegen keine vier Hundertstel auseinander, und eine Gerade schlägt
die meisten davon. Die Bohnenmaße sind glatte, korrelierte Geometrie, also bleibt
den flexiblen Modellen wenig nichtlineare Struktur, aus der sie Nutzen ziehen
könnten.

Wechselt der Datensatz, wechselt die Reihenfolge. Auf **California Housing**
drückt Gradient Boosting den RMSE von 0,7263 bei der linearen Regression auf
**0,4668**, eine Verbesserung um 36 %, weil dieses Problem *voller*
Wechselwirkungen steckt.

**Welches Verfahren gewinnt, ist eine Eigenschaft deiner Daten und keine
Rangliste der Algorithmen.** Das ist der Faden, der durch das ganze Buch läuft,
und mehrere Notebooks hier berichten ein Ergebnis, bei dem ich das Gegenteil
erwartet hatte, und sagen das auch so.

---

## Loslegen

```bash
git clone https://github.com/ELounissi/open-ml-notebooks
cd open-ml-notebooks
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

Alles läuft auf einer **Laptop-CPU in Minuten**. Eine GPU beschleunigt die
Deep-Learning-Kapitel und wird nie vorausgesetzt. Kleine Datensätze liegen im
Repository, deshalb laufen die meisten Notebooks sofort und offline.

**Kompletter Anfänger?** Fang bei [Teil 1, Grundlagen](../01-foundations/) an.
**Kennst du die Grundlagen?** Spring zu dem Verfahren, das du brauchst: jedes Notebook steht für sich.
**Vorbereitung auf Bewerbungsgespräche?** Der Abschnitt "Wann es gewinnt, wann es
verliert" ist in jedem Notebook für genau dieses Gespräch geschrieben.

---

## Weiterverwenden

Der Code steht unter **MIT**, Text und Abbildungen unter **CC BY 4.0**. Nimm sie
mit in einen Kurs, einen Vortrag oder eine Lerngruppe, mit Namensnennung.
Datensätze behalten ihre eigenen Lizenzen, festgehalten pro Datensatz in
[`data/README.md`](../data/README.md).

Wenn dir das Zeit gespart hat, **hilft ein Stern anderen dabei, es zu finden**.
Fehler gefunden? Mach ein Issue auf: Korrekturen sind wirklich willkommen,
besonders solche, die zeigen, dass eine Behauptung hier falsch ist.

---

### Themen

`machine-learning` `deep-learning` `reinforcement-learning` `machine-learning-tutorial`
`jupyter-notebook` `python` `scikit-learn` `pytorch` `data-science` `neural-network`
`cnn` `rnn` `lstm` `transformer` `linear-regression` `logistic-regression`
`random-forest` `xgboost` `clustering` `pca` `learn-machine-learning`
`machine-learning-algorithms` `ml-tutorial` `data-science-tutorial` `education`

`#MachineLearning` `#DeepLearning` `#ReinforcementLearning` `#DataScience`
`#Python` `#ScikitLearn` `#PyTorch` `#LearnMachineLearning` `#MLTutorial`
`#JupyterNotebook` `#OpenSource` `#AI`

---

**Von Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)
