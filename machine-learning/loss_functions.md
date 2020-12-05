# 常见回归和分类损失函数比较
损失函数的定义为$L(y,f(x))$，衡量真实值$y$和预测值$f(x)$ 之间不一致的程度，一般越小越好。为了便于不同损失函数的比较，常将其表示为单变量的函数，在回归问题中这个变量为$y-f(x)$，在分类问题中则为$yf(x)$。下面分别进行讨论。

## 回归问题的损失函数
回归问题中$y$和 $f(x)$ 皆为实数 $\\in R$，因此用残差 $y-f(x)$ 来度量二者的不一致程度。残差（的绝对值）越大，则损失函数越大，学习出来的模型效果就越差（这里不考虑正则化问题）。

__常见的回归损失函数有__：
+ **平方损失 (squared loss)** ：
  > $(y-f(x))^2$
+ **绝对值 (absolute loss)** :
  > $|y-f(x)|$
+ **Huber损失 (huber loss)** :
  > $\left\{\begin{matrix}\frac12[y-f(x)]^2 \qquad |y-f(x)| \leq \delta \\ \delta|y-f(x)| - \frac12\delta^2 \qquad |y-f(x)|> \delta\end{matrix}\right.$

其中最常用的是平方损失，然而其缺点是对于异常点会施以较大的惩罚，因而不够robust。如果有较多异常点，则绝对值损失表现较好，但绝对值损失的缺点是在$y-f(x)=0$处不连续可导，因而不容易优化。

Huber损失是对二者的综合，当$|y-f(x)|$小于一个事先指定的值$\\delta$时，变为平方损失，大于$\\delta$时，则变成类似于绝对值损失，因此也是比较robust的损失函数。三者的图形比较如下：

![](/assert/Regression.png)

***
## 分类问题的损失函数
对于二分类问题，$y\in \left\{-1,+1 \right\}$，损失函数常表示为关于$yf(x)$的单调递减形式。如下图：

![](/assert/Monotone_Decreasing.png)

$yf(x)$ 被称为**margin**，其作用类似于回归问题中的残差$y-f(x)$。

二分类问题中的分类规则通常为 $sign(f(x)) = \left\{\begin{matrix} +1 \qquad if\;\;f(x) \geq 0 \\ -1 \qquad if\;\;f(x) < 0\end{matrix}\right.$

可以看到如果$yf(x) > 0$，则样本分类正确，$yf(x) < 0$ 则分类错误，而相应的分类决策边界即为$f(x) = 0$。所以最小化损失函数也可以看作是最大化margin的过程，任何合格的分类损失函数都应该对margin<0的样本施以较大的惩罚。

### 1、 0-1损失 (zero-one loss)
$$L(y,f(x)) = \left\{\begin{matrix} 0 \qquad if \;\; yf(x)\geq0 \\ 1 \qquad if \;\; yf(x) < 0\end{matrix}\right.$$

0-1损失对每个错分类点都施以相同的惩罚，这样那些“错的离谱“ (即 $margin \rightarrow -\infty$)的点并不会收到大的关注，这在直觉上不是很合适。另外0-1损失不连续、非凸，优化困难，因而常使用其他的代理损失函数进行优化。

### 2、Logistic loss
$$L(y,f(x)) = log(1+e^{-yf(x)})$$

logistic Loss为Logistic Regression中使用的损失函数，下面做一下简单证明:

Logistic Regression中使用了Sigmoid函数表示预测概率:
$$g(f(x)) = P(y=1|x) = \frac{1}{1+e^{-f(x)}}$$
而
$$P(y=-1|x) = 1-P(y=1|x) = 1-\frac{1}{1+e^{-f(x)}} = \frac{1}{1+e^{f(x)}} = g(-f(x))$$

因此利用$y\in\left\{-1,+1\right\}$，可写为$P(y|x) = \frac{1}{1+e^{-yf(x)}}$，此为一个概率模型，利用极大似然的思想：

$$max(\prod P(y|x)) = max(\prod \frac{1}{1+e^{-yf(x)}})$$
两边取对数，又因为是求损失函数，则将极大转为极小：
$$max(\sum logP(y|x)) = -min(\sum log(\frac{1}{1+e^{-yf(x)}})) = min(\sum log(1+e^{-yf(x)}) $$
这样就得到了logistic loss。


如果定义$t = \frac{y+1}2 \in \left\{0,1\right\}$，则极大似然法可写为：
$$\prod (P(y=1|x))^{t}((1-P(y=1|x))^{1-t}$$
取对数并转为极小得：

$$\sum [-t\log P(y=1|x) - (1-t)\log (1-P(y=1|x))]$$
上式被称为交叉熵损失 (cross entropy loss)，可以看到在二分类问题中logistic loss和交叉熵损失是等价的，二者区别只是标签y的定义不同。

### 3、Hinge loss
$$L(y,f(x)) = max(0,1-yf(x))$$
hinge loss为svm中使用的损失函数，hinge loss使得$yf(x)>1$的样本损失皆为0，由此带来了稀疏解，使得svm仅通过少量的支持向量就能确定最终超平面。

### 4、指数损失(Exponential loss)
$$L(y,f(x)) = e^{-yf(x)}$$
exponential loss为AdaBoost中使用的损失函数，使用exponential loss能比较方便地利用加法模型推导出AdaBoost算法 (具体推导过程可见)。然而其和squared loss一样，对异常点敏感，不够robust。

### 5、modified Huber loss
$$ L(y,f(x)) = \left \{\begin{matrix} max(0,1-yf(x))^2 \qquad if \;\;yf(x)\geq-1 \\ \qquad-4yf(x) \qquad\qquad\;\; if\;\; yf(x)&lt;-1\end{matrix}\right.\qquad$$
modified huber loss结合了hinge loss和logistic loss的优点，既能在$yf(x) > 1$时产生稀疏解提高训练效率，又能进行概率估计。另外其对于 $(yf(x) < -1)$ 样本的惩罚以线性增加，这意味着受异常点的干扰较少，比较robust。scikit-learn中的SGDClassifier同样实现了modified huber loss。




最后来张全家福：

![](/assert/Classification.png)

从上图可以看出上面介绍的这些损失函数都可以看作是0-1损失的单调连续近似函数，而因为这些损失函数通常是凸的连续函数，因此常用来代替0-1损失进行优化。它们的相同点是都随着$margin \rightarrow -\infty$而加大惩罚；不同点在于，logistic loss和hinge loss都是线性增长，而exponential loss是以指数增长。

值得注意的是上图中modified huber loss的走向和exponential loss差不多，并不能看出其robust的属性。其实这和算法时间复杂度一样，成倍放大了之后才能体现出巨大差异：

![](/assert/Classification_2.png)


## 问题
1. 平方误差损失函数和交叉熵损失函数分别适合什么场景？
> 如果学习模型致力于解决的问题是回归问题的连续变量，那么使用平方损失函数较为合适；若是对于分类问题的离散Ont-Hot向量，那么交叉熵损失函数较为合适。
>+ 直观理解
> 从平方损失函数运用到多分类场景下，可知平方损失函数对每一个输出结果都十分看重，而交叉熵损失函数只对正确分类的结果看重。平方损失函数除了让正确分类尽量变大，还会让错误分类都变得更加平均，但分类问题中后面的这个调整使没必要的。但是对于回归问题这样的考虑就显得重要了，因而回归问题上使用交叉熵并不适合。

>+ 理论角度分析
> 平方数损失函数假设最终结果都服从高斯分布，而高斯分布实际上是一个连续变量，并不是一个离散变量。如果假设结果变量服从均值 $u$，方差为 $\sigma$，那么利用最大似然法就可以优化它的负对数似然，如下，除去与$y$无关的项目，最后剩下的就是平方损失函数的形式。  
![](/assert/max_likelihood_gaussian.png)
