# CTR 预估 传统模型
<center><img src="/assert/ctr.png"/></center>

***
## 协同过滤 （Collaborative Filtering，CF）
见 [Collaborative Filtering](/machine-learning/collaborative_filtering.md)

***
## LR -> FM -> FFM
### LR与多项式模型
见 [Logistic Regression](/machine-learning/logistic_regression.md)  

假设讨论互联网广告场景下的广告点击率预估问题，给定年龄、性别、教育程度、兴趣类目等用户侧特征，上下文特征，以及广告id，商品类型等广告侧特征，预测用户对于该广告的点击概率 $p(label=1 | user, context, ad)$。
针对特征的表示，通常做法是对各类特征进行one-hot编码，对于类目型或离散型数值特征很好理解；对于连续型数值特征，例如收入，一般会按区间分段方式将其离散化再进行one-hot编码。

每一维特征是相互独立地作用在回归值上，即每一维特征与label的相关性是被单独衡量的。而根据实际样本数据观察，我们会发现对某些特征进行组合以后，组合特征会与label有更强的相关性。要引入这种组合特征信号，一种直观的想法是多项式模型，讨论二项式的情况，如下：
$$y(X)=\omega_0+\sum_{i=1}^{n}{\omega_ix_i}+\sum_{i=1}^{n-1}{\sum_{j=i+1}^{n}{\omega_{ij}x_ix_j}}$$

其中， $n$ 代表样本的特征数量，$x_i$ 是第 $i$ 个特征的值， $w_0,w_i,w_{ij}$ 是模型参数。
从公式来看，模型前半部分就是普通的LR线性组合，后半部分的交叉项即：特征的组合。单从模型表达能力上来看，FM的表达能力是强于LR的，至少不会比LR弱，当交叉项参数全为0时退化为普通的LR模型。

组合特征的参数一共有 $\frac{n(n-1)}{2}$ 个，任意两个参数都是独立的。然而，在数据稀疏性普遍存在的实际应用场景中，二次项参数的训练是很困难的。其原因是：每个参数 $w_{ij}$ 的训练需要大量 $x_i$ 和 $x_j$都非零的样本；由于样本数据本来就比较稀疏，满足“$x_i$ 和 $x_j$都非零”的样本将会非常少。训练样本的不足，很容易导致参数 $w_{ij}$ 不准确，最终将严重影响模型的性能，由此引出FM。

总结
>+ LR是广义线性模型，每个特征都是独立的，如果需要考虑特征与特征之间的相互作用，需要人工对特征进行交叉组合。
>+ 多项式模型（核函数选择为二阶多项式的SVM）可以对特征进行核变换，但是在特征高度稀疏的情况下，并不能很好的进行学习。


### 因子分解机（Factorization Machines，FM）
由矩阵分解中的隐因子模型（Latent Factor Model），我们可以通过隐向量分解来解决高维稀疏问题。

**FM模型的核心思想**

所有二次项参数 $w_{ij}$ 可以组成一个对称阵 $W$ （为了方便说明FM的由来，对角元素可以设置为正实数），那么这个矩阵就可以分解为 $W=V^TV$ ， $V$ 的第 $j$ 列( $v_j$ )便是第 $j$ 维特征( $x_j$ )的隐向量。换句话说，特征分量 $x_i$ 和 $x_j$ 的交叉项系数就等于 $x_i$ 对应的隐向量与 $x_j$ 对应的隐向量的内积，即每个参数 $w_{ij}=⟨v_i,v_j⟩$。

**FM的模型方程**

$$\hat{y}(X):=\omega_0+\sum_{i=1}^{n}{\omega_ix_i}+\sum_{i=1}^{n-1}{\sum_{j=i+1}^{n}{<v_i,v_j>x_ix_j}}$$

$$其中<v_i,v_j>:=\sum_{f=1}^{k}{v_{i,f}\cdot v_{j,f}}$$

隐向量的长度为 $k(k<<n)$ ，包含 $k$ 个描述特征的因子。二次项的参数数量减少为 $kn$ 个，远少于多项式模型的参数数量。参数因子化使得 $x_hx_i$ 的参数和 $x_ix_j$ 的参数不再是相互独立的，所有包含“ $x_i$ 的非零组合特征”（存在某个 $j\ne i$ ，使得 $x_ix_j\ne 0$ ）的样本都可以用来学习隐向量$v_i$，这很大程度上避免了数据稀疏性造成的影响。而在多项式模型中， $w_{hi}$ 和 $w_{ij}$ 是相互独立的，因此我们可以在样本稀疏的情况下相对合理地估计FM的二次项参数。另外，利用隐向量，我们可以评估未曾出现过的交叉特征。

FM是一个通用的拟合方程，可以采用不同的损失函数用于解决回归、二元分类等问题，比如可以采用MSE（Mean Square Error）损失函数来求解回归问题，也可以采用Hinge/Cross-Entropy 损失来求解分类问题。

**FM的计算优化**  

直观上看，FM的复杂度是 $O(kn^2)$ 。但是，通过公式(3)的等式，FM的二次项可以化简，其复杂度可以优化到 $O(kn)$ 。
$$\begin{align*} \sum_{i=1}^{n-1}{\sum_{j=i+1}^{n}{<v_i,v_j>x_ix_j}} &= \frac{1}{2}\sum_{i=1}^{n}{\sum_{j=1}^{n}{<v_i,v_j>x_ix_j}} - \frac{1}{2} {\sum_{i=1}^{n}{<v_i,v_i>x_ix_i}} \\ &= \frac{1}{2} \left( \sum_{i=1}^{n}{\sum_{j=1}^{n}{\sum_{f=1}^{k}{v_{i,f}v_{j,f}x_ix_j}}} - \sum_{i=1}^{n}{\sum_{f=1}^{k}{v_{i,f}v_{i,f}x_ix_i}} \right) \\ &= \frac{1}{2}\sum_{f=1}^{k}{\left[ \left( \sum_{i=1}^{n}{v_{i,f}x_i} \right) \cdot \left( \sum_{j=1}^{n}{v_{j,f}x_j} \right) - \sum_{i=1}^{n}{v_{i,f}^2 x_i^2} \right]} \\ &= \frac{1}{2}\sum_{f=1}^{k}{\left[ \left( \sum_{i=1}^{n}{v_{i,f}x_i} \right)^2 - \sum_{i=1}^{n}{v_{i,f}^2 x_i^2} \right]} \end{align*}$$
    > 解释：  
    $v_{if}$ 是一个具体的值；  
    第1个等号：对称矩阵 $W$ 对角线上半部分；  
    第2个等号：把向量内积 $<v_i,v_j>$ 展开成累加和的形式；  
    第3个等号：提出公共部分；  
    第4个等号： $v_i$ 和 $v_j$ 相当于是一样的，表示成平方过程。  

**FM的参数求解**  

采用随机梯度下降法SGD求解参数
$$\begin{equation} \frac{\partial \hat{y}(x) }{\partial \theta} = \left\{ \begin{array}{lr} 1, & if\ \theta\ is\ \omega_0 \\ x_i, & if\ \theta\ is\ \omega_i\\ x_i\sum_{j=1}^{n}{v_{j,f}x_j - v_{i,f}x_i^2} & if\ \theta\ is\ v_{i,f} \end{array} \right. \end{equation} $$
由上式可知，$v_{i,f}$ 的训练只需要样本的 $x_i$ 特征非0即可，适合于稀疏数据

在使用SGD训练模型时，在每次迭代中，只需计算一次 $\sum_{j=1}^{n}{v_{j,f}x_j}$ ，就能够方便得到所有 $v_{i}$ 的梯度，其时间复杂度是 $O(kn)$ ，模型参数一共有 $nk + n + 1$ 个。因此，FM参数训练的复杂度也是 $O(kn)$。综上可知，FM可以在线性时间训练和预测，是一种非常高效的模型。

**关于隐向量 $V$**  
这里的 $v_i$ 是 $x_i$ 特征的低纬稠密表达，实际中隐向量的长度通常远小于特征维度N。FM学到的隐向量可以看做是特征的一种embedding表示，把离散特征转化为Dense Feature，这种Dense Feature还可以后续和DNN来结合，作为DNN的输入，事实上用于DNN的CTR也是这个思路来做的。


**FM的实际问题**  
1. 如何处理连续特征？
2. FM的输入数据是怎样的？
2. 增量训练中出现新的特征值怎么办？
3. 新增的特征值的indexing会变化吗？
4. index一直增加参数一直变多怎么办？

### Field-aware Factorization Machines, FFM
**核心思想**  

FFM通过引入field的概念，FFM把相同性质的特征归于同一个field。同一个categorical特征经过One-Hot编码生成的数值特征都可以放到同一个field，包括用户性别、职业、品类偏好等。在FFM中，每一维特征 $x_i$，针对其它特征的每一种field $f_j$，都会学习一个隐向量 $v_{i,f_j}$。因此，隐向量不仅与特征相关，也与field相关。例如，“日期”特征在于“地点”特征和“类别”特征进行交叉是使用不同的隐向量，这与“地点”特征和“类别”特征的内在差异相符，也是FFM中“field-aware”的由来。

**模型方程**  
$$\hat{y}(X):=\omega_0+\sum_{i=1}^{n}{\omega_ix_i}+\sum_{i=1}^{n-1}{\sum_{j=i+1}^{n}{<v_{i,f_j},v_{j,f_i}>x_ix_j}}$$
其中，$f_j$ 是第 $j$ 个特征所属的field。如果隐向量的长度为 $k$，那么FFM的二次参数有 $nfk$ 个，远多于FM模型的 $nk$ 个。此外，由于隐向量与field相关，FFM二次项并不能够化简，其预测复杂度是 $O(kn^2)$。

**训练细节** 
1. 样本归一化。FFM默认是进行样本数据的归一化，即 $pa.norm$ 为真；若此参数设置为假，很容易造成数据inf溢出，进而引起梯度计算的$nan$错误。因此，样本层面的数据是推荐进行归一化的。

2. 特征归一化。CTR/CVR模型采用了多种类型的源特征，包括数值型和categorical类型等。但是，categorical类编码后的特征取值只有0或1，较大的数值型特征会造成样本归一化后categorical类生成特征的值非常小，没有区分性。例如，一条用户-商品记录，用户为“男”性，商品的销量是5000个（假设其它特征的值为零），那么归一化后特征“sex=male”（性别为男）的值略小于0.0002，而“volume”（销量）的值近似为1。特征“sex=male”在这个样本中的作用几乎可以忽略不计，这是相当不合理的。因此，将源数值型特征的值归一化到 [0,1] 是非常必要的。

3. 省略零值特征。从FFM模型的表达式可以看出，零值特征对模型完全没有贡献。包含零值特征的一次项和组合项均为零，对于训练模型参数或者目标值预估是没有作用的。因此，可以省去零值特征，提高FFM模型训练和预测的速度，这也是稀疏样本采用FFM的显著优势。

**实际应用优化**
引入field的概念之后，参数量明显增大，导致计算量过大，实际使用中可进行优化。
1. 模型上的优化：双线性FFM 
$$\hat{y}(X):=\omega_0+\sum_{i=1}^{n}{\omega_ix_i}+\sum_{i=1}^{n-1}{\sum_{j=i+1}^{n}{<v_{i,f_j},v_{j,f_i}>x_ix_j}}$$
双线性FFM的二次项和FM类似，仍然是将所有特征映射到同一隐空间。但不同于直接对隐向量做点积，为了增强模型的表达能力，双线性FFM在FM基础上增加了 $k×k$ 参数矩阵 $W$。

这里有三种不同的类型的双线性，第一种是所有特征共享一个矩阵 $W$，引入的参数量为 $k×k$；第二种是一个Field一个 $W_i$，引入的参数量为 $f×k×k$；第三种是一个Field组合一个$W_{ij}$，引入的参数量为 $f×f×k×k$。此处是第一种。
<center><img src="/assert/bilinear-ffm.png"/></center>

2. 业务上的优化：对特征（即field）进行聚类，减小field的数量。

### 参数规模与时间复杂度比较
<center><img src="/assert/lr-fm-ffm.png"/></center>

***
## GBDT+LR
见 [GBDT+LR](/machine-learning/gbdt_lr.md) 

## 参考资料
1. https://zhuanlan.zhihu.com/p/37963267
2. https://mp.weixin.qq.com/s/hmmUi-6y_WwWv6dyPm-EfQ
3. 《深度学习推荐系统》
4. [深入FFM原理与实践-美团](https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html)
