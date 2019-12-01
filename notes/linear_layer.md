# Linear Layer 实现笔记

关键字： **matrix multiplication**;  **外积**; 

## 整体思路

因为全连接层前向和后向过程主要通过乘法和加法来进行计算，可以将全连接的计算过程用矩阵相乘来统一处理。

**前向**：

目标值：

* 当前层的计算结果`y`

通过两个计算过程：

1. $ y = W^{T[i]}x $
2. $ y = y + b^{[i]} $

$y$：表示当前的层的运算结果
$x$：表示当前层的输入值
$W^{T[i]}$：权重矩阵的转置
$b$：表示偏置值
$i$：表示第i层

$W^{T[i]}$的维度： `output_size` * `input_size` (与batch size无关)
$x$的维度：`input_size` * `batch_size`
$y$的维度：`output_size` * `batch_size` 

**反向**：

目标值：

* $\frac{\partial L}{\partial x}$ 输入值的梯度
* $\frac{\partial L}{\partial W}$ 权重的梯度 
* $\frac{\partial L}{\partial b}$ 偏置的梯度

$\frac{\partial L}{\partial y}$ 表示由$i+1$传来的梯度 

$$
\frac{\partial L}{\partial x} = W^{T} \frac{\partial L}{\partial y} 
$$ 



$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y}x^{T}  
$$ 



$$
\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} 
$$ 



### 并行性

 矩阵运算结果的中的元素是相互独立的，所以结果中的每个元素可以并行计算


### kernel函数设计

#### 输出



#### 过程



### 线程组织

```
threads = (BLOCK_DIM_1D)
grid = (target->size() + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D
```

### 优化细节

共享内存

外积




