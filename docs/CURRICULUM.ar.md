<div dir="rtl" markdown="1">

# المنهج

ستة وثمانون دفترًا، واثنا عشر جزءًا، ولكل دفتر طريقة واحدة. اقرأها بالترتيب أو
اذهب مباشرة إلى الذي تحتاجه: كل دفتر قائم بذاته.

كل دفتر يتّبع الهيكل الخماسي نفسه، فما إن تقرأ واحدًا حتى تعرف أين تنظر في
البقية:

1. **الفكرة**: ما تفعله الطريقة، بلغة بسيطة، ومعها رسمة واحدة
2. **الرياضيات**: بالقدر الذي تحتاجه فقط، مكتوبةً بلا مواربة
3. **من الصفر**: تنفيذ مصغَّر بـ`NumPy`، حتى لا يبقى في الأمر سحر
4. **في الممارسة**: نسخة `scikit-learn` أو `PyTorch`، مشروحة سطرًا سطرًا
5. **متى تنجح، ومتى تفشل**: على مجموعات بيانات الكتاب، ومعها السبب

كل دفتر في القوائم أدناه منجَز: منفَّذ، ومعه رسومه وصفحته الخاصة.

---

## الجزء 1: الأساسيات

المفردات التي يقوم عليها كل ما بعدها. ابدأ من هنا إن كنت جديدًا.

| # | الدفتر |
|---|---|
| 01-01 | [ما الذي يفعله التعلّم الآلي فعلًا: مسار العمل من أوله إلى آخره](../01-foundations/01-what-machine-learning-does/) |
| 01-02 | [التدريب والتحقق والاختبار: لماذا تحتاج إلى الثلاثة](../01-foundations/02-train-validation-test/) |
| 01-03 | [الإفراط في التعلّم والقصور فيه، مرئيَّين لا موصوفَين](../01-foundations/03-overfitting-and-underfitting/) |
| 01-04 | [التحقق المتقاطع، وأي صيغة منه تستعمل ومتى](../01-foundations/04-cross-validation/) |
| 01-05 | [مقاييس التصنيف: الدقة (accuracy)، الإحكام (precision)، الاستدعاء (recall)، F1، ROC-AUC](../01-foundations/05-classification-metrics/) |
| 01-06 | [مقاييس الانحدار: MAE، RMSE، R²، MAPE](../01-foundations/06-regression-metrics/) |
| 01-07 | [توحيد مقاييس الخصائص وترميز المتغيّرات الفئوية](../01-foundations/07-feature-scaling-and-encoding/) |
| 01-08 | [البيانات الناقصة: ماذا تفعل، وكم يكلّفك ذلك](../01-foundations/08-missing-data/) |
| 01-09 | [ضبط المعاملات الفائقة: البحث الشبكي والعشوائي والبايزي](../01-foundations/09-hyperparameter-tuning/) |

## الجزء 2: الانحدار

| # | الدفتر |
|---|---|
| 02-01 | [الانحدار الخطي، من المعادلة السويّة إلى scikit-learn](../02-regression/01-linear-regression/) |
| 02-02 | [النزول التدرّجي، مُراقَبًا خطوةً خطوة](../02-regression/02-gradient-descent/) |
| 02-03 | [الانحدار متعدد الحدود ومقايضة الانحياز والتباين](../02-regression/03-polynomial-regression/) |
| 02-04 | [انحدار Ridge (L2)](../02-regression/04-ridge-regression/) |
| 02-05 | [انحدار Lasso (L1) وانتقاء الخصائص تلقائيًا](../02-regression/05-lasso-regression/) |
| 02-06 | [Elastic Net](../02-regression/06-elastic-net/) |
| 02-07 | [الانحدار المقاوم للقيم الشاذة: Huber وRANSAC](../02-regression/07-outlier-resistant-regression/) |
| 02-08 | [الانحدار الكمّي: التنبؤ بمجال، لا برقم](../02-regression/08-quantile-regression/) |
| 02-09 | [بواسون وسائر النماذج الخطية المعمّمة](../02-regression/09-generalised-linear-models/) |

## الجزء 3: التصنيف

| # | الدفتر |
|---|---|
| 03-01 | [الانحدار اللوجستي](../03-classification/01-logistic-regression/) |
| 03-02 | [أقرب الجيران](../03-classification/02-k-nearest-neighbours/) |
| 03-03 | [بايز الساذج](../03-classification/03-naive-bayes/) |
| 03-04 | [التحليل التمييزي الخطي والتربيعي](../03-classification/04-discriminant-analysis/) |
| 03-05 | [آلات المتجهات الداعمة وحيلة النواة](../03-classification/05-support-vector-machines/) |
| 03-06 | [أشجار القرار، وكيف يُختار الانقسام](../03-classification/06-decision-trees/) |
| 03-07 | [الفئات غير المتوازنة: إعادة المعاينة والأوزان والعتبات](../03-classification/07-imbalanced-classes/) |
| 03-08 | [استراتيجيات التصنيف متعدد الفئات ومتعدد الوسوم](../03-classification/08-multiclass-and-multilabel/) |
| 03-09 | [معايرة الاحتمالات](../03-classification/09-probability-calibration/) |

## الجزء 4: التجميعات

| # | الدفتر |
|---|---|
| 04-01 | [Bagging، ولماذا ينفع أخذ المتوسط](../04-ensembles/01-bagging/) |
| 04-02 | [الغابات العشوائية](../04-ensembles/02-random-forest/) |
| 04-03 | [Extra Trees](../04-ensembles/03-extra-trees/) |
| 04-04 | [AdaBoost](../04-ensembles/04-adaboost/) |
| 04-05 | [التعزيز التدرّجي من مبادئه الأولى](../04-ensembles/05-gradient-boosting/) |
| 04-06 | [مقارنة بين XGBoost وLightGBM وCatBoost](../04-ensembles/06-xgboost-lightgbm-catboost/) |
| 04-07 | [Stacking والتصويت](../04-ensembles/07-stacking-and-voting/) |

## الجزء 5: التعلّم غير الموجَّه

| # | الدفتر |
|---|---|
| 05-01 | [k-Means](../05-unsupervised/01-k-means/) |
| 05-02 | [اختيار k: الكوع، مقياس silhouette، إحصاءة الفجوة](../05-unsupervised/02-choosing-k/) |
| 05-03 | [التجميع الهرمي والمخططات الشجرية](../05-unsupervised/03-hierarchical-clustering/) |
| 05-04 | [DBSCAN وHDBSCAN](../05-unsupervised/04-dbscan-and-hdbscan/) |
| 05-05 | [نماذج الخلائط الغاوسية](../05-unsupervised/05-gaussian-mixture-models/) |
| 05-06 | [كشف الشذوذ: Isolation Forest وOne-Class SVM وLOF](../05-unsupervised/06-anomaly-detection/) |
| 05-07 | [قواعد الاقتران بخوارزمية Apriori](../05-unsupervised/07-association-rules/) |

## الجزء 6: تقليل الأبعاد

| # | الدفتر |
|---|---|
| 06-01 | [تحليل المكوّنات الرئيسية](../06-dimensionality-reduction/01-principal-component-analysis/) |
| 06-02 | [Kernel PCA وICA وNMF](../06-dimensionality-reduction/02-kernel-pca-ica-nmf/) |
| 06-03 | [t-SNE](../06-dimensionality-reduction/03-t-sne/) |
| 06-04 | [UMAP](../06-dimensionality-reduction/04-umap/) |
| 06-05 | [انتقاء الخصائص: بالترشيح، بالتغليف، بالتضمين](../06-dimensionality-reduction/05-feature-selection/) |

## الجزء 7: الشبكات العصبية

| # | الدفتر |
|---|---|
| 07-01 | [البيرسبترون](../07-neural-networks/01-the-perceptron/) |
| 07-02 | [البيرسبترون متعدد الطبقات والانتشار العكسي، بـNumPy](../07-neural-networks/02-mlp-and-backpropagation/) |
| 07-03 | [الشبكة نفسها في PyTorch](../07-neural-networks/03-the-same-net-in-pytorch/) |
| 07-04 | [دوال التنشيط ولماذا تهمّ](../07-neural-networks/04-activation-functions/) |
| 07-05 | [المُحسِّنات: SGD وMomentum وRMSProp وAdam](../07-neural-networks/05-optimisers/) |
| 07-06 | [التنظيم: dropout، batch norm، weight decay، الإيقاف المبكر](../07-neural-networks/06-regularisation/) |
| 07-07 | [حلقة تدريب تصلح لإعادة الاستعمال](../07-neural-networks/07-a-training-loop/) |

## الجزء 8: الرؤية الحاسوبية

| # | الدفتر |
|---|---|
| 08-01 | [الالتفاف والتجميع، وما الذي يتعلّمه المرشّح](../08-computer-vision/01-convolution-and-pooling/) |
| 08-02 | [شبكة التفافية على Fashion-MNIST، طبقةً طبقة](../08-computer-vision/02-a-cnn-layer-by-layer/) |
| 08-03 | [معماريات كلاسيكية: LeNet وVGG وResNet](../08-computer-vision/03-classic-architectures/) |
| 08-04 | [زيادة البيانات](../08-computer-vision/04-data-augmentation/) |
| 08-05 | [نقل التعلّم والضبط الدقيق](../08-computer-vision/05-transfer-learning/) |
| 08-06 | [تجزئة الصور، مقدّمة](../08-computer-vision/06-image-segmentation/) |
| 08-07 | [الكشف عن الأجسام، مقدّمة](../08-computer-vision/07-object-detection/) |

## الجزء 9: التسلسلات واللغة

| # | الدفتر |
|---|---|
| 09-01 | [المعالجة الأولية للنصوص، حقيبة الكلمات، وTF-IDF](../09-sequences-and-language/01-text-preprocessing-and-tfidf/) |
| 09-02 | [تضمينات الكلمات: Word2Vec وGloVe](../09-sequences-and-language/02-word-embeddings/) |
| 09-03 | [الشبكات العصبية المتكرّرة](../09-sequences-and-language/03-recurrent-neural-networks/) |
| 09-04 | [LSTM](../09-sequences-and-language/04-lstm/) |
| 09-05 | [GRU، وكيف يقارَن بـLSTM](../09-sequences-and-language/05-gru/) |
| 09-06 | [من تسلسل إلى تسلسل، مع الانتباه](../09-sequences-and-language/06-seq2seq-with-attention/) |
| 09-07 | [المحوّل Transformer، مبنيًّا من الصفر](../09-sequences-and-language/07-the-transformer/) |
| 09-08 | [الضبط الدقيق لمحوّل مدرَّب مسبقًا](../09-sequences-and-language/08-fine-tuning/) |
| 09-09 | [التنبؤ بالسلاسل الزمنية: ARIMA في مواجهة التعلّم الآلي في مواجهة التعلّم العميق](../09-sequences-and-language/09-time-series-forecasting/) |

## الجزء 10: النماذج التوليدية

| # | الدفتر |
|---|---|
| 10-01 | [المرمّزات التلقائية](../10-generative-models/01-autoencoders/) |
| 10-02 | [المرمّزات التلقائية التبايُنية (VAE)](../10-generative-models/02-variational-autoencoders/) |
| 10-03 | [الشبكات التوليدية التنافسية](../10-generative-models/03-generative-adversarial-networks/) |
| 10-04 | [نماذج الانتشار، أصغر مثال يعمل](../10-generative-models/04-diffusion-models/) |

## الجزء 11: التعلّم المعزّز

| # | الدفتر |
|---|---|
| 11-01 | [إطار التعلّم المعزّز: الوكلاء والحالات والمكافآت](../11-reinforcement-learning/01-the-setup/) |
| 11-02 | [قطّاع الطرق متعدّدو الأذرع](../11-reinforcement-learning/02-multi-armed-bandits/) |
| 11-03 | [عمليات القرار الماركوفية، والتكرار على القيمة وعلى السياسة](../11-reinforcement-learning/03-markov-decision-processes/) |
| 11-04 | [Q-learning](../11-reinforcement-learning/04-q-learning/) |
| 11-05 | [SARSA، وفيمَ يختلف عن Q-learning](../11-reinforcement-learning/05-sarsa/) |
| 11-06 | [شبكات Q العميقة](../11-reinforcement-learning/06-deep-q-networks/) |
| 11-07 | [تدرّجات السياسة وREINFORCE](../11-reinforcement-learning/07-policy-gradients/) |
| 11-08 | [الممثّل والناقد وPPO](../11-reinforcement-learning/08-actor-critic-and-ppo/) |

## الجزء 12: تجميع الخيوط

| # | الدفتر |
|---|---|
| 12-01 | [**جدول النتائج**: كل طريقة على كل مجموعة من مجموعات بيانات الكتاب](../12-putting-it-together/01-the-scoreboard/) |
| 12-02 | [تفسير النماذج: أهمية التبديل، SHAP، LIME](../12-putting-it-together/02-interpreting-models/) |
| 12-03 | [خطوط المعالجة، وألّا تتسرّب البيانات بعد اليوم](../12-putting-it-together/03-pipelines/) |
| 12-04 | [حفظ النموذج وتحميله وتقديمه](../12-putting-it-together/04-saving-and-serving/) |
| 12-05 | [الأخطاء التي يقع فيها الجميع](../12-putting-it-together/05-common-mistakes/) |

---

## مجموعات بيانات الكتاب

كل الدفاتر تستعمل المجموعة الصغيرة نفسها، حتى تعني المقارنة بين فصل وآخر شيئًا.
ترى فورًا أين يختلف SVM عن الغابة العشوائية، لأن السؤال المطروح عليهما كان
واحدًا.

| مجموعة البيانات | المهمة | الحجم | لماذا هذه تحديدًا |
|---|---|---|---|
| California Housing | انحدار | 20,640 × 8 | تأتي مع `scikit-learn`، فالفصل الأول يعمل دون أي تنزيل |
| Breast Cancer Wisconsin | تصنيف ثنائي | 569 × 30 | صغيرة إلى حدّ أن كل طريقة تتدرّب عليها في لحظة |
| **UCI Dry Bean** | تصنيف إلى 7 فئات | 13,611 × 16 | نُشرت عام 2020 ونادرًا ما تُستعمل في الدروس، وهذا يُخرج الكتاب عن الطريق المطروق |
| UCI Bike Sharing | انحدار، سلاسل زمنية | 17,379 × 16 | مجموعة واحدة تصلح للانحدار الجدولي ولنماذج التسلسلات معًا |
| Fashion-MNIST | تصنيف صور | 70,000 × 28 × 28 | بشكل MNIST نفسه لكنها أصعب بمرات، فلا تأتي نتائج الشبكات الالتفافية كلها عند 99% |

المصادر والرخص وتواريخ الجلب في [`data/README.md`](../data/README.md).

## جدول النتائج

يجمع الجزء 12 نتيجة كل طريقة على كل مجموعة من مجموعات بيانات الكتاب في جدول
واحد، ثم يشرح النمط. بعضه متوقَّع وبعضه ليس كذلك: التعزيز التدرّجي يتصدّر عادةً
على البيانات الجدولية، وأقرب الجيران ينهار كلما تكاثرت الأعمدة، والنموذج الخطي
يتفوّق على الشبكة العصبية حين تكون الصفوف قليلة. المقصود من الكتاب أن تنهيه وأنت
تعرف *لماذا*، لا *أيّها* فقط.

---

من إعداد **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

---

## الاستفادة منه، ومساعدته على الوصول

الكود تحت رخصة MIT. انسخه وعدّله وضعه في مشاريعك أو في تدريسك، دون إذن ودون
إلزام بذكر المصدر. إن وفّر عليك فصلٌ منه بعد ظهيرة كاملة، فهذا ما وُجد له.

إن وجدته مفيدًا، فإن **نجمة واحدة تساعد غيرك على العثور عليه**، وهي تقريبًا
الطريقة الوحيدة لينتشر كتاب كهذا. التصحيحات والاعتراضات مرحَّب بها في issues،
وخاصةً إن أعدت تشغيل شيء وخرجت بنتيجة مختلفة.

---

</div>
