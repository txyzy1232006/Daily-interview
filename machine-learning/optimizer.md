# 优化算法

## 梯度下降法、牛顿法和拟牛顿法

1. 为什么梯度相反的方向是函数下降最快的方向？

## 从 SGD 到 Adam [zhihu](https://zhuanlan.zhihu.com/p/32626442)

**Gradient Descent** 

(1) 计算目标函数关于参数的梯度
$$g_t = \nabla_\theta J(\theta)$$
(2) 根据历史梯度计算一阶和二阶动量
$$m_t = \phi(g_1, g_2, \cdots, g_t)$$
$$v_t = \psi(g_1, g_2, \cdots, g_t)$$
(3) 更新模型参数
$$\theta_{t+1} = \theta_t - \frac{1}{\sqrt{v_t + \epsilon}} m_t$$
其中，$\epsilon$ 为平滑项，防止分母为零，通常取 1e-8。

**Vanilla SGD**
$$m_t = \eta g_t\\v_t = I^2 \\ \epsilon = 0$$
$$\theta_{i+1}= \theta_t - \eta g_t$$

**SGD-M**  

加上动量
$$m_t = \gamma m_{t-1} + \eta g_t$$

**Nesterov Accelerated Gradient, NGA**  

计算梯度时减去一阶动量
$$g_t = \nabla_\theta J(\theta - \gamma m_{t-1})$$

**Adagrad**

引入二阶动量，之前所有时刻梯度的平方和
$$v_t = \text{diag}(\sum_{i=1}^t g_{i,1}^2, \sum_{i=1}^t g_{i,2}^2, \cdots, \sum_{i=1}^t g_{i,d}^2)$$
其中， $v_t \in \mathbb{R}^{d\times d}$ 是对角矩阵，其元素 $v_{t, ii}$ 为参数第 $i$ 维从初始时刻到时刻 $t$ 的梯度平方和。

**RMSprop**

二阶动量只考虑最近的梯度，指数移动平均
$$v_t = \gamma v_{t-1} + (1-\gamma) \cdot \text{diag}(g_t^2)$$

**Adam**

一阶动量也使用指数移动平均
$$m_t = \eta[ \beta_1 m_{t-1} + (1 - \beta_1)g_t ]$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) \cdot \text{diag}(g_t^2)$$
其中，初值 $m_0 = 0, v_0=0$。  
注意到，在迭代初始阶段，$m_t$ 和 $v_t$ 有一个向初值的偏移（过多的偏向了 0）。因此，可以对一阶和二阶动量做偏置校正 (bias correction)
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1-\beta_2^t}$$
则有
$$\theta_{t+1} = \theta_t - \frac{1}{\sqrt{\hat{v}_t} + \epsilon } \hat{m}_t$$

**NAdam**

在 Adam 之上融合了 NAG 的思想
$$g_t=\nabla f(w_t-\alpha \cdot m_{t-1} / \sqrt{v_t + \epsilon})$$


## 牛顿法
泰勒二阶展开
