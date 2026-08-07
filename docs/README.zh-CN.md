[English](../README.md) · [العربية](README.ar.md) · [Français](README.fr.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [Português](README.pt.md) · [हिन्दी](README.hi.md) · [简体中文](README.zh-CN.md)

# open-ml-notebooks

### 读代码学机器学习

一本免费开放的书，**86 个 Jupyter 笔记本**，覆盖机器学习、深度学习和强化学习：
一个笔记本讲一个方法，先用大白话讲清楚，再从零手写一遍，最后用库里正规的写法
做一次。

每个笔记本都**已经运行好了**。在 GitHub 上随便打开一个，图、数字和输出就在那里。
想开始读，什么都不用装。

笔记本本身是英文的，这不构成障碍：代码、变量名、打印出来的表格和图表标签都是
英文，而这本来就是所有机器学习库和所有报错信息使用的语言。图看起来是一样的，
数字也是一样的，不用能流利读英文散文，也能一路跟下每一个结果。

**作者：[Elyes Lounissi](https://www.linkedin.com/in/elyes-lounissi/)**

![一条直线看不见两个通勤高峰](../02-regression/01-linear-regression/figures/fig-04-where-it-loses.png)

*出自[线性回归](../02-regression/01-linear-regression/)：为什么最简单的模型会在
周期性数据上失手，以及把 0.39 拉到 0.68 的那一行修改。*

---

## 为什么写这本书

大多数教程告诉你该调哪个函数。很少有教程讲清楚这个方法到底在做什么，几乎没有
教程告诉你**它什么时候是错的选择**。

这里的每个笔记本都回答四个问题：

1. **这个方法到底在做什么？**先用文字讲，配一张图。
2. **数学是什么？**写出来，只写你用得上的那部分。
3. **我能自己写一遍吗？**一个最小的 NumPy 版本，让一切都不再是黑箱。
4. **它什么时候赢，什么时候输？**用测量说话，不靠断言。

第四个问题最要紧，也正是通常被略过的那个。

## 把整本书串起来的那个想法

每个笔记本都用**同样的五个数据集**。这是故意的。第三部分里支持向量机拿到
0.93，第四部分里随机森林拿到 0.94，这两个数字可以直接比较，因为问的是同一个
问题。

| 数据集 | 任务 | 规模 | 为什么选它 |
|---|---|---|---|
| California Housing | 回归 | 20,640 × 8 | scikit-learn 自带：第一部分不用下载任何东西就能跑 |
| Breast Cancer Wisconsin | 二分类 | 569 × 30 | 小到每个方法都能瞬间训练完 |
| **UCI Dry Bean** | 7 类分类 | 13,611 × 16 | 2020 年才发布，教程里几乎没人用，而且不平衡得恰到好处 |
| UCI Bike Sharing | 回归 + 时间序列 | 17,379 × 16 | 一个数据集同时服务表格数据和序列这两类章节 |
| Fashion-MNIST | 图像分类 | 70,000 张图像 | 形状和 MNIST 一样，难上好几倍，所以 CNN 那几章不会索然无味 |

全书最后，[**总成绩榜**](CURRICULUM.zh-CN.md)把每个方法在每个数据集上的结果放进
同一张表，并解释其中的规律：为什么表格数据通常被梯度提升拿下，为什么列一多
k 近邻就垮掉，为什么行数很少时线性模型能赢过神经网络。

---

## 目录

全部 86 个笔记本的完整目录：**[CURRICULUM.zh-CN.md](CURRICULUM.zh-CN.md)**

| 部分 | 内容 |
|---|---|
| **1：[基础](../01-foundations/)** | 工作流程、训练／验证／测试、过拟合、交叉验证、评价指标、缩放、编码、缺失值、调参 |
| **2：[回归](../02-regression/)** | 线性回归、梯度下降、多项式回归、Ridge、Lasso、Elastic Net、Huber、RANSAC、分位数回归、广义线性模型 |
| **3：[分类](../03-classification/)** | 逻辑回归、k 近邻、朴素贝叶斯、LDA／QDA、SVM 与核方法、决策树、类别不平衡、概率校准 |
| **4：[集成方法](../04-ensembles/)** | Bagging、随机森林、Extra Trees、AdaBoost、梯度提升、XGBoost、LightGBM、CatBoost、Stacking |
| **5：[无监督学习](../05-unsupervised/)** | k-Means、怎么选 k、层次聚类、DBSCAN、HDBSCAN、高斯混合模型、异常检测、Apriori |
| **6：[降维](../06-dimensionality-reduction/)** | PCA、核 PCA、ICA、NMF、t-SNE、UMAP、特征选择 |
| **7：[神经网络](../07-neural-networks/)** | 感知机、用 NumPy 写多层感知机与反向传播、PyTorch、激活函数、优化器、正则化 |
| **8：[计算机视觉](../08-computer-vision/)** | 卷积、CNN、LeNet／VGG／ResNet、数据增强、迁移学习、图像分割、目标检测 |
| **9：[序列与语言](../09-sequences-and-language/)** | TF-IDF、词嵌入、RNN、LSTM、GRU、seq2seq、注意力机制、Transformer、时间序列 |
| **10：[生成模型](../10-generative-models/)** | 自编码器、VAE、GAN、扩散模型 |
| **11：[强化学习](../11-reinforcement-learning/)** | 多臂老虎机、MDP、Q-learning、SARSA、DQN、策略梯度、Actor-Critic、PPO |
| **12：[汇总](../12-putting-it-together/)** | 总成绩榜、SHAP 与 LIME、Pipeline、部署、常见错误 |

### 先打开这几个

- **[01-03：过拟合与欠拟合](../01-foundations/03-overfitting-and-underfitting/)**，
  机器学习里最有用的那一个诊断，是画出来的而不是定义出来的；里面还有一个 20 次
  多项式拟合，它不只在测试集上比 4 次的差，在训练集上也更差，因为设计矩阵的条件数
  达到 1.1e+21，求解器还没等到过拟合发生就悄悄放弃了
- **[02-01：线性回归](../02-regression/01-linear-regression/)**，推导正规方程，
  用 30 行 NumPy 写的版本和 scikit-learn 对到小数点后 12 位，残差诊断揭出汇总
  统计量掩盖掉的偏差，以及这个方法在周期性数据上失手之后，靠编码而不是靠更大的
  模型被救回来
- **[02-04：岭回归](../02-regression/04-ridge-regression/)**，L2 惩罚项，以及一个
  诚实的结论：在这份数据上它**一点精度都没换来**，却让系数**稳定了 32 倍**；
  另外还有一个结果反过来的缩放实验，它自己解释了自己
- **[02-05：Lasso 回归](../02-regression/05-lasso-regression/)**，为什么一个指数
  就能把收缩变成选择，以及一个发现：**`LassoCV` 把 30 个纯噪声列里的 17 个留了
  下来**，因为交叉验证优化的是预测，不是稀疏性
- **[03-01：逻辑回归](../03-classification/01-logistic-regression/)**，sigmoid、
  对数损失、手写出来的梯度下降，在七种数量不均的豆子上用 softmax，以及为什么最
  *稀有*的那一类反倒是最好认的
- **[03-02：k 近邻](../03-classification/02-k-nearest-neighbours/)**，一个完全不
  训练的模型；光是做缩放就值 **+0.205 的准确率**，`k=1` 在训练数据上拿到毫无意义
  的满分 1.000，维数灾难在这里是被测出来的，不是被断言的
- **[03-03：朴素贝叶斯](../03-classification/03-naive-bayes/)**，为什么一个人人都
  知道是假的假设照样管用，以及关于 scikit-learn 的 `var_smoothing` 默认值的十三点
  绕行，这个默认值会悄悄毁掉小尺度的特征
- **[03-05：支持向量机](../03-classification/05-support-vector-machines/)**，间隔、
  `C` 究竟在控制什么，以及把两个圆环抬进第三个维度来演示核技巧，然后解释为什么你
  根本不需要真的把它建出来；再加上没人提的那笔开销：27 倍的行数要花 **139 倍的
  时间**
- **[03-06：决策树](../03-classification/06-decision-trees/)**，从零写切分搜索，
  用算术找出整份数据里最好的那一个问题，一棵读得懂的深度为 3 的树，以及一棵放开
  长的树，它在训练集上达到满分准确率，在留出数据上却*更差*
- **[04-02：随机森林](../04-ensembles/02-random-forest/)**，解释整套设计的方差
  公式、袋外评分、两个互相打架的特征重要性指标，以及和逻辑回归的正面对决，森林
  **没有赢**
- **[04-05：梯度提升](../04-ensembles/05-gradient-boosting/)**，为什么拟合残差
  *就是*梯度下降、学习率上的取舍，以及和森林最尖锐的区别：树加到一定程度，提升
  反而会**变差**
- **[05-01：k-Means](../05-unsupervised/01-k-means/)**，从零实现 Lloyd 算法；在
  真实簇数已知的数据上，肘部法和轮廓系数一起指向了*错误*的簇数；还有四种失效
  模式，包括 k-means 信心十足地把纯噪声切成整整齐齐的几组
- **[05-04：DBSCAN 与 HDBSCAN](../05-unsupervised/04-dbscan-and-hdbscan/)**，密度
  聚类能找到 k-means 找不到的月牙形，用 k 距离曲线的拐点来定 `eps` 而不是靠猜，
  以及一个发现：**没有任何 `eps` 取值会告诉你「这份数据没有结构」**
- **[06-01：PCA](../06-dimensionality-reduction/01-principal-component-analysis/)**，
  手算特征分解与 scikit-learn 吻合到 2.2×10⁻¹⁶，几个主成分最后被解释成「大小」和
  「形状」，以及一个诚实的结论：比起简单地保留全部列，PCA **从来没有赢过**
- **[07-01：感知机](../07-neural-networks/01-the-perceptron/)**，一个神经元、不
  需要微积分的学习规则，以及让神经网络停滞了十年的 XOR 那四个点
- **[07-02：多层感知机与反向传播](../07-neural-networks/02-mlp-and-backpropagation/)**，
  每一个梯度都手工推导，并**用数值方法核对到 1.9×10⁻¹⁰**，然后只用 NumPy 搭出一个
  能跑的网络
- **[07-03：同一个网络的 PyTorch 版](../07-neural-networks/03-the-same-net-in-pytorch/)**，
  在一个能手算核对的例子上演示 autograd、那个你会一直用下去的五行训练循环，以及
  每个人都会犯一次的三个错误
- **[08-02：逐层拆解一个 CNN](../08-computer-vision/02-a-cnn-layer-by-layer/)**，
  讲清楚权重共享，以及一个卷积网络在准确率、参数量**和**速度上都胜过全连接网络；
  它第一层的卷积核自己变成了没人要求过的边缘检测器
- **[11-04：Q-learning](../11-reinforcement-learning/04-q-learning/)**，不依赖
  `gym`、从零搭起来的 Cliff Walking，一行写完的 Bellman 更新，以及那个经典结果：
  Q-learning 找到更好的策略，SARSA 却收到更多的奖励

这些加在一起覆盖了有监督回归、有监督分类、集成方法、无监督聚类、降维和强化
学习，而且每一个都已经写完并运行过。

### 总成绩榜

因为每个笔记本用的是同一批数据集，比较会一点点累积起来。在 **UCI Dry Bean**
上，5 折交叉验证的准确率：

| 方法 | 准确率 |
|---|---|
| SVM，RBF 核 | **0.9301** |
| 梯度提升 | 0.9271 |
| SVM，线性核 | 0.9262 |
| 随机森林 | 0.9244 |
| 逻辑回归 | 0.9234 |
| k 近邻 | 0.9231 |
| 神经网络（NumPy，从零实现） | 0.9306 * |
| 朴素贝叶斯 | 0.8972 |
| 决策树 | 0.8945 |

\* 这一行用的是单次留出划分，不是 5 折，所以不能直接比较。我为什么不把它算作
一场胜利，见对应的[笔记本](../07-neural-networks/02-mlp-and-backpropagation/)。

九个方法彼此相差不到 0.04，而一条直线赢过其中大多数。豆子的测量值是平滑且彼此
相关的几何量，灵活的模型没有多少非线性结构可以利用。

换一个数据集，排名就变了。在 **California Housing** 上，梯度提升把 RMSE 从线性
回归的 0.7263 降到 **0.4668**（提升 36%），因为那个问题里*确实*到处都是交互
作用。

**哪个方法会赢，是你数据的属性，不是算法的排行榜。**这条线贯穿全书；书里有好
几个笔记本报告的结果和我原本的预期相反，而我把这一点直说了出来。

---

## 开始使用

```bash
git clone https://github.com/ELounissi/open-ml-notebooks
cd open-ml-notebooks
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

所有内容都能在**普通笔记本电脑的 CPU 上几分钟内跑完**。GPU 会让深度学习那几章
更快，但从来不是必需的。小数据集已经提交进仓库，所以大多数笔记本可以立刻离线
运行。

**完全是新手？**从[第 1 部分，基础](../01-foundations/)开始。
**已经懂基础？**直接跳到你需要的那个方法：每个笔记本都能独立阅读。
**在准备面试？**每个笔记本里「什么时候赢，什么时候输」那一节，写的就是面试里
会聊到的东西。

---

## 拿去用

代码是 **MIT**。文字和图表是 **CC BY 4.0**。可以把它们带进课程、演讲或读书小组，
注明出处即可。数据集各自保留原有许可，逐个记录在
[`data/README.md`](../data/README.md) 里。

如果这些内容帮你省了时间，**点个 star 能让更多人找到它**。发现错误了？开一个
issue：欢迎指正，尤其欢迎那种能证明书里某个说法是错的指正。

---

### 标签

`machine-learning` `deep-learning` `reinforcement-learning` `machine-learning-tutorial`
`jupyter-notebook` `python` `scikit-learn` `pytorch` `data-science` `neural-network`
`cnn` `rnn` `lstm` `transformer` `linear-regression` `logistic-regression`
`random-forest` `xgboost` `clustering` `pca` `learn-machine-learning`
`machine-learning-algorithms` `ml-tutorial` `data-science-tutorial` `education`

`#MachineLearning` `#DeepLearning` `#ReinforcementLearning` `#DataScience`
`#Python` `#ScikitLearn` `#PyTorch` `#LearnMachineLearning` `#MLTutorial`
`#JupyterNotebook` `#OpenSource` `#AI`

---

**作者：Elyes Lounissi** ·
[LinkedIn](https://www.linkedin.com/in/elyes-lounissi/) ·
[pilot.tun@gmail.com](mailto:pilot.tun@gmail.com)
