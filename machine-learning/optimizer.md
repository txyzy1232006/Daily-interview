# 优化算法

## 梯度下降法、牛顿法和拟牛顿法

1. 为什么梯度相反的方向是函数下降最快的方向？

## 从 SGD 到 Adam [zhihu](https://zhuanlan.zhihu.com/p/32626442)
SGD
SGD-M：加上动量
NGA：计算梯度时减去一阶动量
Adagrad：引入二阶动量，之前所有时刻梯度的平方和
RMSprop：二阶动量只考虑最近的梯度，指数移动平均
Adam：一阶动量也使用指数移动平均
NAdam：计算梯度时用一阶和二阶动量
