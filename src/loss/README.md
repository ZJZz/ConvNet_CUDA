# Loss 实现笔记

关键字： **reduction**

## 整体思路

因为是分类问题，所以采用*Cross Entropy*作为*Loss*函数

$$
L = -\sum_{i=0}^m y^{(i)}\log \hat{y}^{(i)} = -\sum_{i=0}^m y^{(i)} \log(softmax(o^{i}))
$$

$m$：表示类别个数
$y^{(i)}$：表示第$i$类的预测值
$\hat{y}^{(i)}$：表示第$i$类的真实值
$o{(i)}$：表示最后一层，即全连接层，对类别$i$的输出



### 并行性

* 因为一个*batch*中的图片之间的*loss*是独立的
* 一张图片所属类别的 $y^{(i)}\log \hat{y}^{(i)}$ 的计算是相互独立的

所以可以让每个线程计算一个batch中一张图片的所有类别的$y^{(i)}\log \hat{y}^{(i)}$，最后使用并行reduction来进行求和。

### kernel函数设计

#### 目标量

一个batch的loss

#### 过程

* 确定是当前线程是batch中第几个图片`batch_idx`，因为需要这个值来计算这张图片的真实标签`target`和预测值`predict`的偏移量
* 计算这张图片的所有10个类别的$y^{(i)}\log \hat{y}^{(i)}$ ，累加起来得到单张图片的`loss`，拷贝到暂存工作区`workspace`
* 因为不同线程块之间无法同步，所以使用1个线程块计算整个batch的loss, 先把其它线程块中的值拷贝到线程块`blockIdx=0`的共享内存中，用累加的方式进行，因为每个图片的loss已经算完，使用结合律不影响加法的最终结果
*  使用交错配对的规约方式来进行reduction

### 执行配置

```
threads = (BLOCK_DIM_1D)
grid = (target->size() + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D
```

### 优化细节






