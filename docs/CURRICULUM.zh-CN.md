# 课程大纲

86 个笔记本，12 个部分，一个笔记本讲一个方法。可以按顺序读，也可以直接跳到你
需要的那一个：每个笔记本都能独立阅读。

每个笔记本都是同样的五段结构，所以读完一个，你就知道在其余每一个里该往哪里找：

1. **想法**：这个方法做什么，用大白话讲，配一张图
2. **数学**：只写你用得上的部分，写出来，不含糊其辞
3. **从零实现**：一个最小的 NumPy 实现，让一切都不再是黑箱
4. **实际用法**：scikit-learn 或 PyTorch 的版本，逐行注解
5. **什么时候赢，什么时候输**：在固定数据集上，并给出原因

下面每个笔记本都已经完成：运行过，带着自己的图表和自己的页面。

---

## 第 1 部分：基础

后面一切都会用到的那套词汇。如果你是新手，从这里开始。

| # | 笔记本 |
|---|---|
| 01-01 | [机器学习到底在做什么：从头到尾的完整流程](../01-foundations/01-what-machine-learning-does/) |
| 01-02 | [训练集、验证集、测试集：为什么三个都要](../01-foundations/02-train-validation-test/) |
| 01-03 | [过拟合与欠拟合，看见而不是听说](../01-foundations/03-overfitting-and-underfitting/) |
| 01-04 | [交叉验证，以及什么时候该用哪一种](../01-foundations/04-cross-validation/) |
| 01-05 | [分类的评价指标：准确率、精确率、召回率、F1、ROC-AUC](../01-foundations/05-classification-metrics/) |
| 01-06 | [回归的评价指标：MAE、RMSE、R²、MAPE](../01-foundations/06-regression-metrics/) |
| 01-07 | [特征缩放与类别变量编码](../01-foundations/07-feature-scaling-and-encoding/) |
| 01-08 | [缺失值：怎么处理，代价是什么](../01-foundations/08-missing-data/) |
| 01-09 | [超参数调优：网格搜索、随机搜索与贝叶斯搜索](../01-foundations/09-hyperparameter-tuning/) |

## 第 2 部分：回归

| # | 笔记本 |
|---|---|
| 02-01 | [线性回归：从正规方程到 scikit-learn](../02-regression/01-linear-regression/) |
| 02-02 | [梯度下降：一步一步看着它走](../02-regression/02-gradient-descent/) |
| 02-03 | [多项式回归与偏差和方差的权衡](../02-regression/03-polynomial-regression/) |
| 02-04 | [岭回归（L2）](../02-regression/04-ridge-regression/) |
| 02-05 | [Lasso 回归（L1）与自动特征选择](../02-regression/05-lasso-regression/) |
| 02-06 | [Elastic Net 弹性网络](../02-regression/06-elastic-net/) |
| 02-07 | [抗离群点的回归：Huber 与 RANSAC](../02-regression/07-outlier-resistant-regression/) |
| 02-08 | [分位数回归：预测一个区间，而不是一个数](../02-regression/08-quantile-regression/) |
| 02-09 | [泊松回归与其他广义线性模型](../02-regression/09-generalised-linear-models/) |

## 第 3 部分：分类

| # | 笔记本 |
|---|---|
| 03-01 | [逻辑回归](../03-classification/01-logistic-regression/) |
| 03-02 | [k 近邻](../03-classification/02-k-nearest-neighbours/) |
| 03-03 | [朴素贝叶斯](../03-classification/03-naive-bayes/) |
| 03-04 | [线性判别分析与二次判别分析](../03-classification/04-discriminant-analysis/) |
| 03-05 | [支持向量机与核技巧](../03-classification/05-support-vector-machines/) |
| 03-06 | [决策树，以及一个切分是怎么选出来的](../03-classification/06-decision-trees/) |
| 03-07 | [类别不平衡：重采样、权重与阈值](../03-classification/07-imbalanced-classes/) |
| 03-08 | [多分类与多标签的处理策略](../03-classification/08-multiclass-and-multilabel/) |
| 03-09 | [概率校准](../03-classification/09-probability-calibration/) |

## 第 4 部分：集成方法

| # | 笔记本 |
|---|---|
| 04-01 | [Bagging，以及为什么求平均会有帮助](../04-ensembles/01-bagging/) |
| 04-02 | [随机森林](../04-ensembles/02-random-forest/) |
| 04-03 | [极端随机树（Extra Trees）](../04-ensembles/03-extra-trees/) |
| 04-04 | [AdaBoost](../04-ensembles/04-adaboost/) |
| 04-05 | [梯度提升：从最基本的原理讲起](../04-ensembles/05-gradient-boosting/) |
| 04-06 | [XGBoost、LightGBM、CatBoost 三者对比](../04-ensembles/06-xgboost-lightgbm-catboost/) |
| 04-07 | [Stacking 与投票集成](../04-ensembles/07-stacking-and-voting/) |

## 第 5 部分：无监督学习

| # | 笔记本 |
|---|---|
| 05-01 | [k-Means 聚类](../05-unsupervised/01-k-means/) |
| 05-02 | [怎么选 k：肘部法、轮廓系数、间隔统计量](../05-unsupervised/02-choosing-k/) |
| 05-03 | [层次聚类与树状图](../05-unsupervised/03-hierarchical-clustering/) |
| 05-04 | [DBSCAN 与 HDBSCAN](../05-unsupervised/04-dbscan-and-hdbscan/) |
| 05-05 | [高斯混合模型](../05-unsupervised/05-gaussian-mixture-models/) |
| 05-06 | [异常检测：孤立森林、One-Class SVM、LOF](../05-unsupervised/06-anomaly-detection/) |
| 05-07 | [用 Apriori 挖掘关联规则](../05-unsupervised/07-association-rules/) |

## 第 6 部分：降维

| # | 笔记本 |
|---|---|
| 06-01 | [主成分分析](../06-dimensionality-reduction/01-principal-component-analysis/) |
| 06-02 | [核 PCA、ICA 与 NMF](../06-dimensionality-reduction/02-kernel-pca-ica-nmf/) |
| 06-03 | [t-SNE](../06-dimensionality-reduction/03-t-sne/) |
| 06-04 | [UMAP](../06-dimensionality-reduction/04-umap/) |
| 06-05 | [特征选择：过滤式、包裹式、嵌入式](../06-dimensionality-reduction/05-feature-selection/) |

## 第 7 部分：神经网络

| # | 笔记本 |
|---|---|
| 07-01 | [感知机](../07-neural-networks/01-the-perceptron/) |
| 07-02 | [用 NumPy 实现多层感知机与反向传播](../07-neural-networks/02-mlp-and-backpropagation/) |
| 07-03 | [同一个网络的 PyTorch 版](../07-neural-networks/03-the-same-net-in-pytorch/) |
| 07-04 | [激活函数，以及它们为什么重要](../07-neural-networks/04-activation-functions/) |
| 07-05 | [优化器：SGD、Momentum、RMSProp、Adam](../07-neural-networks/05-optimisers/) |
| 07-06 | [正则化：Dropout、批归一化、权重衰减、早停](../07-neural-networks/06-regularisation/) |
| 07-07 | [一个可以反复复用的训练循环](../07-neural-networks/07-a-training-loop/) |

## 第 8 部分：计算机视觉

| # | 笔记本 |
|---|---|
| 08-01 | [卷积、池化，以及一个卷积核学到了什么](../08-computer-vision/01-convolution-and-pooling/) |
| 08-02 | [逐层拆解一个跑在 Fashion-MNIST 上的 CNN](../08-computer-vision/02-a-cnn-layer-by-layer/) |
| 08-03 | [经典架构：LeNet、VGG、ResNet](../08-computer-vision/03-classic-architectures/) |
| 08-04 | [数据增强](../08-computer-vision/04-data-augmentation/) |
| 08-05 | [迁移学习与微调](../08-computer-vision/05-transfer-learning/) |
| 08-06 | [图像分割入门](../08-computer-vision/06-image-segmentation/) |
| 08-07 | [目标检测入门](../08-computer-vision/07-object-detection/) |

## 第 9 部分：序列与语言

| # | 笔记本 |
|---|---|
| 09-01 | [文本预处理、词袋模型与 TF-IDF](../09-sequences-and-language/01-text-preprocessing-and-tfidf/) |
| 09-02 | [词嵌入：Word2Vec 与 GloVe](../09-sequences-and-language/02-word-embeddings/) |
| 09-03 | [循环神经网络](../09-sequences-and-language/03-recurrent-neural-networks/) |
| 09-04 | [LSTM](../09-sequences-and-language/04-lstm/) |
| 09-05 | [GRU，以及它和 LSTM 的比较](../09-sequences-and-language/05-gru/) |
| 09-06 | [带注意力机制的序列到序列模型](../09-sequences-and-language/06-seq2seq-with-attention/) |
| 09-07 | [从零搭一个 Transformer](../09-sequences-and-language/07-the-transformer/) |
| 09-08 | [微调一个预训练 Transformer](../09-sequences-and-language/08-fine-tuning/) |
| 09-09 | [时间序列预测：ARIMA、机器学习与深度学习三方对比](../09-sequences-and-language/09-time-series-forecasting/) |

## 第 10 部分：生成模型

| # | 笔记本 |
|---|---|
| 10-01 | [自编码器](../10-generative-models/01-autoencoders/) |
| 10-02 | [变分自编码器](../10-generative-models/02-variational-autoencoders/) |
| 10-03 | [生成对抗网络](../10-generative-models/03-generative-adversarial-networks/) |
| 10-04 | [扩散模型：能跑起来的最小例子](../10-generative-models/04-diffusion-models/) |

## 第 11 部分：强化学习

| # | 笔记本 |
|---|---|
| 11-01 | [强化学习的基本设定：智能体、状态、奖励](../11-reinforcement-learning/01-the-setup/) |
| 11-02 | [多臂老虎机](../11-reinforcement-learning/02-multi-armed-bandits/) |
| 11-03 | [马尔可夫决策过程、值迭代与策略迭代](../11-reinforcement-learning/03-markov-decision-processes/) |
| 11-04 | [Q-learning](../11-reinforcement-learning/04-q-learning/) |
| 11-05 | [SARSA，以及它和 Q-learning 的区别](../11-reinforcement-learning/05-sarsa/) |
| 11-06 | [深度 Q 网络](../11-reinforcement-learning/06-deep-q-networks/) |
| 11-07 | [策略梯度与 REINFORCE](../11-reinforcement-learning/07-policy-gradients/) |
| 11-08 | [Actor-Critic 与 PPO](../11-reinforcement-learning/08-actor-critic-and-ppo/) |

## 第 12 部分：汇总

| # | 笔记本 |
|---|---|
| 12-01 | [**总成绩榜**：每个方法在每个固定数据集上的成绩](../12-putting-it-together/01-the-scoreboard/) |
| 12-02 | [解释模型：置换重要性、SHAP、LIME](../12-putting-it-together/02-interpreting-models/) |
| 12-03 | [Pipeline，以及从此不再泄漏数据](../12-putting-it-together/03-pipelines/) |
| 12-04 | [模型的保存、加载与上线](../12-putting-it-together/04-saving-and-serving/) |
| 12-05 | [每个人都会犯的那些错](../12-putting-it-together/05-common-mistakes/) |

---

## 固定数据集

每个笔记本都用同一小组数据集，所以跨章节的比较才是有意义的。你可以一眼看出支持
向量机和随机森林差在哪里，因为它们被问的是同一个问题。

| 数据集 | 任务 | 规模 | 为什么选它 |
|---|---|---|---|
| California Housing | 回归 | 20,640 × 8 | scikit-learn 自带，所以第一部分不用下载任何东西就能跑 |
| Breast Cancer Wisconsin | 二分类 | 569 × 30 | 小到每个方法都能瞬间训练完 |
| **UCI Dry Bean** | 7 类分类 | 13,611 × 16 | 2020 年才发布，教程里几乎没人用，这让整本书不至于走在人人都走过的路上 |
| UCI Bike Sharing | 回归、时间序列 | 17,379 × 16 | 一个数据集既能做表格回归，也能喂给序列模型 |
| Fashion-MNIST | 图像分类 | 70,000 × 28 × 28 | 形状和 MNIST 一样，难上好几倍，所以 CNN 的结果不会清一色都是 99% |

来源、许可和获取日期都记在 [`data/README.md`](../data/README.md) 里。

## 总成绩榜

第 12 部分把每个方法在每个固定数据集上的结果汇进一张表，然后解释其中的规律。
有些在意料之中，有些不在：表格数据上通常是梯度提升领先，列一多 k 近邻就垮掉，
行数很少时线性模型能赢过神经网络。这本书的意义在于，你读完之后知道的是
*为什么*，而不只是*哪一个*。

---

作者 **Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)

---

## 拿去用，也帮它被更多人看到

代码是 MIT 许可。复制它、改它、把它放进你自己的项目或你的教学里，不需要征求
同意，也不需要署名。如果某一章帮你省下一个下午，它本来就是为这个而写的。

如果你觉得有用，**点个 star 能让别人也找到它**，这是这样一本书唯一的传播方式。
欢迎在 issues 里提出指正和不同意见，尤其是当你重跑某个实验、得到了不一样的
答案。

---
