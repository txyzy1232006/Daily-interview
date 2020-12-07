# CTR预估 深度模型
<center><img src="/assert/ctr.png"/></center>

***
## Embedding
<center><img src="/assert/ctr_embedding.png"/></center>
在神经网络模型中，将特征映射成embedding的操作（这种操作也叫做embedding lookup）可以看作是对输入层的原始one-hot（或多值特征对应的multi-hot）特征向量加上全连接线性变换，用数学符号可以表示成特征向量x和参数矩阵W的乘法运算。在点击率预估等任务上，一般以特征field的embedding作为神经网络的embedding层输出，对于单值特征，field embedding等于特征的embedding；对于多值特征，field embedding等于多个特征embedding的求和结果。当然，求和是对多值特征embedding的sum pooling操作，在具体应用上，也可以使用mean/max/k-max pooling代替。


***
## DNN 

#### 经典DNN网络框架

<center><img src="/assert/ctr_dnn.png"/></center>  
  
#### DNN框架下的FNN、PNN与DeepCrossing模型

+ **FNN模型**: 使用预训练的FM隐向量作为DNN第一层全连接层的输入。
<center><img src="/assert/ctr_fnn.png"/></center>
由于FNN的初始embedding是由FM模型预训练得到的，这样的embedding初始化质量要取决于FM模型，引入了一些外部依赖。另外，FNN的z层向量输入到MLP全连接层，接受的是“加法”操作，而PNN论文作者认为这样的“加法”操作可能不如向量“乘法”操作能进一步的建模不同特征filed之间的局部关系。  

+ **PNN模型**: 在embedding层和MLP全连接隐层之间增加了一个乘积层（product layer），用于更直接的建模两两特征的交互作用关系。  
<center><img src="/assert/ctr_pnn.png"/></center>
PNN中乘积层包含z向量和p向量两部分，z向量由常数“1”向量和特征embedding相乘得到，因此z向量部分实际相当于特征embedding的直接拼接。p向量部分是PNN的核心，它是两两特征embedding进行“乘法”操作后的产物。PNN模型，包括IPNN、OPNN、PNN*（将inner product和outer product进行拼接），效果都要优于FNN。另外也能看到，基于深度神经网络的模型，效果普遍优于LR、FM线性模型，这说明非线性高阶特征建模有很重要的提升作用。

+ **DeepCrossing模型**: 引入残差单元，好处是可以使用更深的网络层数，建模更高阶的特征，增强模型的表达能力。
<center><img src="/assert/ctr_deepcrossing.png"/></center>

***
## Wide & Deep
Wide&Deep模型是一种线性模型和深度模型的联合学习框架，线性模型以浅层形式直接学习稀疏组合特征权重，对训练数据中出现过的组合特征具有很好的记忆能力。而深度模型，稀疏特征被映射成低维稠密embedding向量，随后在深层全连接网络中获得充分交互，对具体的特征组合的“记忆”能力会减弱，但换来了更好的泛化效果。
<center><img src="/assert/ctr_wide_and_deep.png"/></center>
模型的左侧是输入单元和输出单元直接连接的线性模型，即方程形式是y=wx+b，输入单元接收的是未经嵌入的高维稀疏交叉特征向量。模型的右侧是一个DNN网络，输入层的稀疏特征先映射成低维稠密embedding，经过拼接后再馈入到隐层全连接网络进行前向计算。最终模型以左侧Wide部分输出和右侧Deep部分输出的加权和，作为预测事件发生的概率函数的变量。

相比传统的ensemble模型集成形式，即各模型独立训练并且每个模型通常需要足够大的参数规模以确保模型的表达能力，Wide&Deep模型中，Wide部分只需考虑加入少量的强交叉特征，用很少的参数来弥补Deep模型记忆力的不足，实现更好的效果。

（一） wide 和 deep 的比较
>+ wide：广泛应用于具有稀疏、大规模场景。组合特征有效且可解释性强，但需要很多特征工程，且对于未出现过的组合无法学习。 **人为选择的特征组合要放到wide/LR一侧，保证被记住**。
>+ deep：需要较少的特征工程，泛化能力强，可以通过稀疏特征 embedding 学习到未出现过的特征组合。但容易过泛化，推荐不太相关的东西。  
>+ wide & deep：记忆和泛化的结合。

（二）memorization 和 generalization（EE问题）
>+ memorization：exploit，学习频繁出现的特征组合，从历史数据中学习相关性。容易推荐和用户浏览历史相似的东西。
>+ generalization：explore，基于相关性的传递，学习未出现过的特征组合。容易推荐不一样的，新的东西。

（三）模型训练
>+ wide：FTRL，有L1正则的作用，可以得到性能较好的稀疏解。**增量学习、在线学习需要保持解的稀疏性**。
>+ deep：AdaGrad


#### Wide部分的改进
+ **DeepFM模型**：Wide部分使用了FM来构建，利用FM自动学习二阶交叉特征的能力，代替需要人工交叉特征的LR。另外，Wide部分和Deep部分的底层输入特征以及特征embedding向量完全共享，能实现端到端学习。
<center><img src="/assert/ctr_deepfm.jpg"/></center>

>+ 是端对端的学习模型，wide 部分和 deep 部分共享一样的输入，不需要额外的特征工程，能够同时学习到低阶和高阶的特征交互。
>+ 线性模型虽然十分有效，但是无法刻画交互特征，需要很多特征工程，缺点是无法刻画高阶特征交互，也无法学习到在训练集中出现次数很少的特征组合。FM能更有效的学习到2阶交互特征，尤其是在稀疏场景下。

+ **Deep&Cross（DCN）模型**：用Cross网络来建模Wide部分，可以用更高效的方式实现更高阶的交叉。
<center><img src="/assert/ctr_dcn.png"/></center>  

交叉迭代公式:
$$x_{t+1} = x_0 x_i^Tw_i + x_i$$

>+ 提出一种新型的交叉网络结构，可以用来提取交叉组合特征，并不需要人为设计的特征工程；
>+ 这种网络结构足够简单同时也很有效，可以获得随网络层数增加而增加的多项式阶（polynomial degree）交叉特征；
>+ 十分节约内存（依赖于正确地实现），并且易于使用；实现n阶需要的网络层数为n-1，对于d维特征，cross网络需要的参数数量为 $d×(n-1)×2$，因此空间复杂度为 $O(dn)$，只需要线性复杂度。
>+ 实验结果表明，DCN相比于其他模型有更出色的效果，与DNN模型相比，较少的参数却取得了较好的效果。


#### Deep部分的改进

+ **NFM模型**：引入Bi-Interaction Pooling层，替代经典DNN的concat层，在底层增加足够的特征交互信息后，再馈入到MLP网络做进一步的高阶非线性建模。Bi-Interaction Pooling将两两特征embedding做哈达玛积交互后，再通过求和的方式压缩成一个向量。经过化简后，上式的计算复杂度可以从 $O(kn2)$ 优化到 $O(kn)$：

+ **AFM模型**：引入了attention权重因子来对交互后的向量进行加权求和压缩，增强了二阶交叉的表达能力。不过，压缩后的向量不再经过MLP，而是直接一层full connection输出到prediction score，这就将模型限制在只表达二阶交叉特征，缺少更高阶信息的建模。


#### 引入新的子网络
+ **xDeepFM模型**：在传统Wide部分和Deep部分之外，提出了一种新的特征交叉子网络，Compressed Interaction Network（CIN），用于显式地以向量级vector-wise方式建模高阶交叉特征。
<center><img src="/assert/ctr_xdeepfm.png"/></center>  

由于包含了线性模块（Wide）、DNN模块（Deep）和CIN模块，xDeepFM模型是同时具备显式和隐式特征交叉的建模能力的。

xDeepFM论文认为，Deep&Cross中的cross网络存在两点不足，1）cross网络是限定在一种特殊形式下实现高阶交叉特征的，即cross网络中每一层的输出都是 $x^0$ 的标量乘积，$α^ix^0$；2）cross网络是以bit-wise形式交叉的，即便同一个field embedding向量下的元素也会相互影响。对此提出了vector-wise的交互的CIN模块。
<center><img src="/assert/ctr_cin.png"/></center>  

上面这张图我们也把CIN第k层的计算拆解成两步，第一步是对 $X^0$ 和 $X^{k-1}$ 中的两两embedding向量做哈达玛积，得到一个中间张量 $Z^k$，这一步实际就是对两两特征做交互，所以称为interaction。第二步是使用Hk个截面（类似CNN中的“kernel”），每个截面对中间张量 $Z^k$ 从上往下扫，每扫一层计算该层所有元素的加权和，一个截面扫完 $D$ 层得到 $X^k$ 的一个embedding向量（类似CNN中的“feature map”），$H^k$ 个截面扫完得到 $H^k$ 个embedding向量。这一步实际是用kernel将中间张量压缩成一个embedding向量，所以我们称此过程为compression。这也是整个子网络被命名为Compressed Interaction Network（CIN）的原因。从这两步计算过程可以看到，特征交互是在vector-wise级别上进行的（哈达玛积），尽管中间经过压缩，每一个隐层仍然保留着embedding vector的形式，所以说CIN实现的是vector-wise的特征交叉。
<center><img src="/assert/ctr_cin2.png"/></center>  

网络的最终输出 $p^+$，是每层的 $X^k$ 沿embedding维度sum pooling后再拼接的向量，维度是所有隐层的feature map个数之和。

***
## 引入注意力机制：提高模型自适应能力与可解释性
注意力机制本质上上就是加权求和池化（weighted sum pooling），权重有多种计算方式，提高模型自适应能力与可解释性，放在MLP之前能起到特征重要性筛选的作用。

#### AFM模型
<center><img src="/assert/ctr_afm.png"/></center>  

$$\bar{y}_{AFM}(x)=w_0+\sum_{i=1}^{n}{w_ix_i}+p^T\sum_{i=1}^{n}{\sum_{j=i+1}^{n}{a_{ij}(v_i\odot v_j)x_ix_j}},\label{eq:poly}$$
模型参数集合为 $\aleph=\left\{ {w_0, {w_i}_{n=1}^{n}, {v_i}_{i=1}^{n},p,W,b,h} \right\}$。

AFM的核心在于
+ pair-wise 交互层：哈达玛积
$$f_{PI}(\varepsilon) =\left\{ (v_i\odot v_j)x_ix_j \right\}_{(i,j)\in R_x}$$
+ attention-based 池化层
$$f_{Att}(f_{PI}(\varepsilon))=\sum_{(i,j)\in R_x}{a_{ij}(v_i\odot v_j)x_ix_j}$$
其中，$a_{ij}$ 是特征组合权重 $w_{ij}$ 的attention score。一个常规的想法就是通过最小化loss来学习，但是存在一个问题是：但对于训练数据中没有共现过的特征们，它们组合的attention分数无法估计。因此论文进一步提出attention network，用多层感知器MLP来参数化attention分数。
$$a_{ij}^{'}=h^T ReLU(W(v_i\odot v_j)x_ix_j+b),\\ a_{ij}=\frac{exp(a_{ij}^{'})}{\sum_{}^{}{exp(a_{ij}^{'})}}\tag{5}\\(i,j)\in R_x$$

其中，$W\in R^{t\times k}$，$b\in R^t$， $h\in R^t$ 是模型参数， $t$ 表示Attention network的隐藏层大小，即attention factor。

可以看到，Attention Network实际上是一个one layer MLP，激活函数使用ReLU，它的输入是两个嵌入向量element-wise product之后的结果（interacted vector，用来在嵌入空间中对组合特征进行编码）；它的输出是组合特征对应的Attention Score。最后，通过softmax函数来归一化attention分数。

AFM防止过拟合
> AFM模型比FM模型有更强的表达能力，更容易过拟合。因此，可以考虑dropout和L2正则化防止过拟合。
>+ 对于Pair-wise Interaction Layer，采用dropout避免co-adaptations
>+ 对于Attention network（one layer MLP），对权重矩阵W使用L2正则化防止过拟合

#### AutoInt模型
2018年提出的AutoInt模型是一个使用多头自注意力机制增强模型解释性，同时又具备高阶交叉建模能力的模型。
<center><img src="/assert/ctr_autoint.jpg"/></center>  

模型结构如上图，主要创新点在interacting layer。它使用经典的 Multi-head Self-Attention 来构造组合特征，即 key-value attention 的实现方式。一层interacting layer的计算过程：
<center><img src="/assert/ctr_autoint2.png"/></center> 

每个Attention head 都对应着三个转换矩阵： $W_{Query}, W_{Key}, W_{Value} \in \mathbb{R}^{d'*d}.$ 对于第 h 个 Attention head，当第 m 个嵌入向量 $e_m$ 作为query时，其对应输出 $\widetilde{e}_m^{(h)}$ 为：
$$\alpha_{m,k}^{(h)} = \frac{ exp( \phi^{(h)} ( e_m, e_k )  ) } { \sum_{l=1}^{M} exp( \phi^{(h)} ( e_m, e_l )  ) } , \\ \phi^{(h)} ( e_m, e_k )  =  <W_{Query}^{(h)} e_m, W_{Key}^{(h)} e_k> , \\ \widetilde{e}_m^{(h)} = \sum_{k=1}^{M}  \alpha_{m,k}^{(h)}  ( W_{Value}^{(h)} e_k )$$
上式中， $\phi(.)$ 是可选的相似度计算函数，文中简单地选择向量内积。注意，在每个Attention head中，每个嵌入向量 $e$都有一次作为query的机会，从而学习到在这个head下的新表达 $\widetilde{e}$ 。

对第m个嵌入 $e_m$ ，作者简单拼接它在 $H$ 个Attention head的输出：
$$\widetilde{e}_m =  \widetilde{e}_m^{(1)} \oplus \widetilde{e}_m^{(2)} \oplus ... \oplus \widetilde{e}_m^{(H)} \in \mathbb{R}^{d'H} , \\ e^{Res}_{m} = Relu~( \widetilde{e}_m + W_{Res} * e_m ), ~~~ W_{Res}  \in \mathbb{R}^{d'H*d}$$
然后引入标准的残差连接作为其最终输出：

最终的预测输出为：
$\hat{y} =  \sigma(w^T  (e^{Res}_1 \oplus e^{Res}_2 \oplus ... \oplus e^{Res}_M ) +b)$，其中 $w \in \mathbb{R}^{d'HM}$ , $\sigma(.)$ 表示sigmoid函数。

文中采用logloss作为损失函数。另外，虽然这里只展示了单层 Interacting Layer，AutoInt 可以叠加多个这样的层，构造更高阶的组合特征。当然，也可以在旁边搭个Deep层一起合作。

#### FiBiNET模型
FiBiNET主要新颖点是加入了SENET Layer和Bilinear-Interaction Layer。
<center><img src="/assert/ctr_fibinet.png"/></center> 

1）SENET Layer：

SENET Layer的主要作用是学习不同特征的重要度，对不同特征向量进行加权。  
包括三个步骤  
   1.  Sequeen，将每个特征组embedding向量 $e_i$ 压缩成标量 $z_i$，用 $z_i$ 表示第i个特征组的全局统计信息。压缩的方法可以是avg-pooling或max-pooling等。
   2.  Excitation，基于压缩后的统计向量Z，学习特征组的重要度权重A，原文使用的是两层神经网络，第一层为维度缩减层，第二层为维度提升层。
   3.   Re-Weight，通过前一步得到的重要度权重A对原始特征组embedding向量进行重新赋权，得到SENET-Like Embeddings。通过引入额外参数矩阵W，先对vi和W进行内积，再与vj进行哈达玛积，增强模型表达能力来学习特征交叉。

2）Bilinear-Interaction Layer：

通过引入额外参数矩阵 $W$ ，先对 $v_i$ 和 $W$ 进行内积，再与 $v_j$ 进行哈达玛积，增强模型表达能力来学习特征交叉。

引入参数W有3种形式，第一种是所有特征共享一个矩阵W，原文称为Field-All Type；第二种是一个Field一个Wi，称为Field-Each Type；第三种是一个Field组合一个Wij，称为Field-Interaction Type。

#### DIN模型

<center><img src="/assert/ctr_din.png"/></center> 

DIN的核心
1. Embedding层之后的attention: Local Activation。DIN把用户特征、用户历史行为特征进行embedding操作，视为对用户兴趣的表示，之后通过attention network，对每个兴趣表示赋予不同的权值。这个权值是由用户的兴趣和待估算的广告进行匹配计算得到的，如此模型结构符合了之前的两个观察——用户兴趣的多样性以及部分对应。注意：其中商品与商品交叉，店铺与店铺交叉。

2. 激活函数：基于PReLU的Dice激活函数，修改了当 $y<0$ 时的整流函数，首先，对 $x$ 进行均值归一化处理，使得整流点是在数据的均值处，其次，经过一个sigmoid函数的计算，得到了一个0到1的概率值。
$$Dice\left( y \right)=p\left( y \right).y+\alpha\left( 1-p\left( y \right) \right)y ~~~~ y < 0 \\ \space p\left( y \right)=\frac{1}{1+e^{-\frac{y-E\left( y \right)}{\sqrt Var\left( y \right)+\epsilon}}}$$
3. 自适应正则：在CTR预估任务中，用户行为数据具有长尾分布的特点，也即数据非常的稀疏。为了防止模型过拟合，论文设计了一个自适应的正则方法。
4. 评价指标：GAUC，计算了用户级别的AUC，在将其按展示次数进行加权，消除了用户偏差对模型评价的影响，更准确地描述了模型对于每个用户的表现效果。

#### DIEN模型
<center><img src="/assert/ctr_dien.png"/></center> 

DIEN的两个主要创新点：

1. 兴趣抽取层（Interest Extractor Layer），利用GRU序列建模，从用户历史行为中抽取兴趣状态表示h(t)，并且引入辅助loss利用下一时刻的行为信息帮助指导当前时刻h(t)的学习，更准确的建模用户兴趣向量。

2. 兴趣演化层（Interest Evolving Layer），引入AUGRU（GRU with attentional update gate）建模用户兴趣的演化过程，在AUGRU的隐状态更新中引入目标广告和兴趣向量的attention因子，目的是使最终输出给DNN网络的兴趣终态 $h'(T)$ 和目标广告有更高的相关度。

#### BST模型 Behavior Sequence Transformer
<center><img src="/assert/ctr_bst.jpg"/></center> 

利用Multi-head Self-attention，捕捉用户行为序列的序列信息。在对User Behavior Sequence做embedding之后加入了Transformer Layer。

***
## 参考
1. https://mp.weixin.qq.com/s/9C0cQ5E7AUPshmgOl6LybA
2. 《深度学习推荐系统》
