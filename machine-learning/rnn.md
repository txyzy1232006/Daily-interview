# 循环神经网络 RNN

## 传统的RNN
#### 结构
![](/assert/rnn.png)

传统的RNN也即BasicRNNcell单元。内部的运算过程为，(t-1)时刻的隐层输出与w矩阵相乘，与t时刻的输入乘以u之后的值进行相加，然后经过一个非线性变化（tanh或Relu），然后以此方式传递给下一个时刻。

传统的RNN每一步的隐藏单元只是执行一个简单的tanh或者Relu操作。

#### 梯度消失和梯度爆炸
RNN 对于长时记忆的困难主要来源于梯度爆炸/消失问题，总的Loss是每个time step的加和：
$$\mathcal{\large{L}} (\hat{\textbf{y}}, \textbf{y}) = \sum_{t = 1}^{T} \mathcal{ \large{L} }(\hat{\textbf{y}_t}, \textbf{y}_{t})$$
由 backpropagation through time (BPTT) 算法，参数的梯度为：
$$\frac{\partial \boldsymbol{\mathcal{L}}}{\partial \textbf{W}} = \sum_{t=1}^{T} \frac{\partial \boldsymbol{\mathcal{L}}_{t}}{\partial \textbf{W}} = \sum_{t=1}^{T} \frac{\partial \boldsymbol{\mathcal{L}}_t}{\partial \textbf{y}_{t}} \frac{\partial \textbf{y}_{t}}{\partial \textbf{h}_{t}} \overbrace{\frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{k}}}^{ \bigstar } \frac{\partial \textbf{h}_{k}}{\partial \textbf{W}} \\$$

其中 $\frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{k}}$ 包含一系列 $\text{Jacobian}$ 矩阵，
$$\frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{k}} = \frac{\partial \textbf{h}_{t}}{\partial \textbf{h}_{t-1}} \frac{\partial \textbf{h}_{t-1}}{\partial \textbf{h}_{t-2}} \cdots \frac{\partial \textbf{h}_{k+1}}{\partial \textbf{h}_{k}}   = \prod_{i=k+1}^{t} \frac{\partial \textbf{h}_{i}}{\partial \textbf{h}_{i-1}} \\$$

由于 RNN 中每个 time step 都是用相同的 $\textbf{W}$，可得
$$\prod_{i=k+1}^{t} \frac{\partial \textbf{h}_{i}}{\partial \textbf{h}_{i-1}} = \prod_{i=k+1}^{t} \textbf{W}^\top \text{diag} \left[ f'\left(\textbf{h}_{i-1}\right) \right] \\$$

由于 $\textbf{W}_{hh} \in \mathbb{R}^{h \times h}$ 为方阵，对其进行特征值分解：
$$\mathbf{W} = \mathbf{V} \, \text{diag}(\boldsymbol{\lambda}) \, \mathbf{V}^{-1} \\$$

由于上式是连乘 $t$ 次 $\textbf{W}$:
$$\mathbf{W}^t = (\mathbf{V} \, \text{diag}(\boldsymbol{\lambda}) \, \mathbf{V}^{-1})^t = \mathbf{V} \, \text{diag}(\boldsymbol{\lambda})^t \, \mathbf{V}^{-1} \\$$

连乘的次数多了之后，则若最大的特征值 $\lambda >1$ ，会产生梯度爆炸； $\lambda <1$ ，则会产生梯度消失 。不论哪种情况，都会导致模型难以学到有用的模式。

#### 梯度爆炸的解决办法
+ Truncated Backpropagation through time：每次只 BP 固定的 time step 数，类似于 mini-batch SGD。缺点是丧失了长距离记忆的能力。
+ 梯度裁剪：当梯度超过一定的 threshold 后，就进行 element-wise 的裁剪，该方法的缺点是又引入了一个新的参数 threshold。同时该方法也可视为一种基于瞬时梯度大小来自适应 learning rate 的方法。


#### 梯度消失的解决办法
+ 使用 LSTM、GRU等升级版 RNN，使用各种 gates 控制信息的流通。
+ 将权重矩阵 $\textbf{W}$ 初始化为正交矩阵。
+ 反转输入序列。像在机器翻译中使用 seq2seq 模型，若使用正常序列输入，则输入序列的第一个词和输出序列的第一个词相距较远，难以学到长期依赖。将输入序列反向后，输入序列的第一个词就会和输出序列的第一个词非常接近，二者的相互关系也就比较容易学习了。


**注意**：即使采用ReLU，经过反向传播后只要W不是单位矩阵， 梯度还是会出现消失或者爆炸的现象。


***
## LSTM
LSTM 中引入了门控机制来控制信息的累计速度，包括有选择地加入新的信息，并有选择地遗忘之前累计的信息。
![](/assert/lstm.png)
$$\begin{align} \text{input gate}&: \quad  \textbf{i}_t = \sigma(\textbf{W}_i\textbf{x}_t + \textbf{U}_i\textbf{h}_{t-1} + \textbf{b}_i)\tag{1} \\ \text{forget gate}&: \quad  \textbf{f}_t = \sigma(\textbf{W}_f\textbf{x}_t + \textbf{U}_f\textbf{h}_{t-1} + \textbf{b}_f) \tag{2}\\ \text{output gate}&: \quad  \textbf{o}_t = \sigma(\textbf{W}_o\textbf{x}_t + \textbf{U}_o\textbf{h}_{t-1} + \textbf{b}_o) \tag{3}\\ \text{new memory cell}&: \quad  \tilde{\textbf{c}}_t = \text{tanh}(\textbf{W}_c\textbf{x}_t + \textbf{U}_c\textbf{h}_{t-1} + \textbf{b}_c) \tag{4}\\ \text{final memory cell}& : \quad \textbf{c}_t =   \textbf{f}_t \odot \textbf{c}_{t-1} + \textbf{i}_t \odot \tilde{\textbf{c}}_t \tag{5}\\ \text{final hidden state} &: \quad \textbf{h}_t= \textbf{o}_t \odot \text{tanh}(\textbf{c}_t) \tag{6} \end{align}$$

式 $(1) \sim (4)$ 的输入都一样，因而可以合并：
$$\begin{pmatrix} \textbf{i}_t \\ \textbf{f}_{t} \\ \textbf{o}_t \\ \tilde{\textbf{c}}_t \end{pmatrix}  =   \begin{pmatrix} \sigma \\ \sigma \\ \sigma \\ \text{tanh} \end{pmatrix}  \left(\textbf{W}  \begin{bmatrix} \textbf{x}_t \\ \textbf{h}_{t-1} \end{bmatrix} + \textbf{b} \right) \\$$

$\tilde{\textbf{c}}_t$ 为时刻 t 的候选状态， $\textbf{i}_t$ 控制 $\tilde{\textbf{c}}_t$ 中有多少新信息需要保存， $\textbf{f}_t$ 控制上一时刻的内部状态 $\textbf{c}_{t-1}$ 需要遗忘多少信息， $\textbf{o}_t$ 控制当前时刻的内部状态 $\textbf{c}_t$ 有多少信息需要输出给外部状态 $\textbf{h}_t$ 。

事实上连乘多个 $\textbf{f}_t$ 同样会导致梯度消失，但是 LSTM 的一个初始化技巧就是将 forget gate 的 bias 置为正数（例如 1 或者 5，如 tensorflow 中的默认值就是 1.0 ），这样一来模型刚开始训练时 forget gate 的值都接近 1，不会发生梯度消失 (反之若 forget gate 的初始值过小则意味着前一时刻的大部分信息都丢失了，这样很难捕捉到长距离依赖关系)。 随着训练过程的进行，forget gate 就不再恒为 1 了。不过，一个训好的模型里各个 gate 值往往不是在 [0, 1] 这个区间里，而是要么 0 要么 1，很少有类似 0.5 这样的中间值，其实相当于一个二元的开关。假如在某个序列里，forget gate 全是 1，那么梯度不会消失；某一个 forget gate 是 0，模型选择遗忘上一时刻的信息。


***
## GRU
在LSTM中 forget gate 和 input gate 是互补关系，因而比较冗余，GRU 将其合并为一个 update gate。同时 GRU 也不引入额外的记忆单元 (LSTM 中的 $\textbf{c}$ ) ，而是直接在当前状态 $\textbf{h}_t$ 和历史状态 $\textbf{h}_{t-1}$ 之间建立线性依赖关系。
![](/assert/GRU.png)
$$\normalsize \begin{align} \text{reset gate}&: \quad  \textbf{r}_t = \sigma(\textbf{W}_r\textbf{x}_t + \textbf{U}_r\textbf{h}_{t-1} + \textbf{b}_r)\tag{7} \\ \text{update gate}&: \quad  \textbf{z}_t = \sigma(\textbf{W}_z\textbf{x}_t + \textbf{U}_z\textbf{h}_{t-1} + \textbf{b}_z)\tag{8} \\ \text{new memory cell}&: \quad  \tilde{\textbf{h}}_t = \text{tanh}(\textbf{W}_h\textbf{x}_t + \textbf{r}_t \odot (\textbf{U}_h\textbf{h}_{t-1}) + \textbf{b}_h) \tag{9}\\ \text{final hidden state}&: \quad \textbf{h}_t = \textbf{z}_t \odot \textbf{h}_{t-1} + (1 - \textbf{z}_t) \odot \tilde{\textbf{h}}_t \tag{10} \end{align}$$

$\tilde{\textbf{h}}_t$ 为时刻 t 的候选状态， $\textbf{o}_t$ 控制 $\tilde{\textbf{h}}_t$ 有多少依赖于上一时刻的状态 $\textbf{h}_{t-1}$ （LSTM中的 $\textbf{o}_t$ ），如果 $\textbf{r}_t = 1$ ，则式 $(9)$ 与 Vanilla RNN 一致。对于短依赖的 GRU 单元，reset gate 通常会更新频繁。 $\textbf{z}_t$ 控制当前的内部状态 $\textbf{h}_t$ 中有多少来自于上一时刻的 $\textbf{h}_{t-1}$ 。如果 $\textbf{z}_t = 1$ ，则会每步都传递同样的信息，和当前输入 $\textbf{x}_t$ 无关。

***
## Seq2Seq
Seq2Seq模型的核心思想是，通过深度神经网络将一个作为输入的序列映射为一个作为输出的序列，这一过程由编码输入与解码输出两个环节构成。在经典的实现中，编码器和解码器各由一个循环神经网络构成，既可以选择传统循环神经网络结构，也可以使用长短期记忆模型、门控循环单元等。在Seq2Seq模型中，两个循环神经网络是共同训练的。

#### 优化解码
+ Beam Search
+ 堆叠的RNN
+ 增加Dropout机制
+ 与编码器之间建立残差连接

***
## cs224n 中提出的 RNN 训练 tips：
![](/assert/rnn_tips.jpg)
