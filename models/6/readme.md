对网络作出调整：

- 论文对数据进行降采样，并且随机选取positive和negative面片
- 我的数据没有做降采样，一个口扫数据大概有300000个面片，其中100000个面片都是positive。所以我的采样方法是对每一类面片随机采样相同个数的点
- 使用了预训练模型
- 效果不是很好，可能是因为dataset1的数据比较粗糙，所以这次只使用dataset2的数据

存在的问题：

- 原论文面片的精度十分粗糙，整个口扫才10000个面片，我们的数据精度很高
- 看懂了A_S和A_L邻接矩阵，但是对于我自己的数据，由于数据精度不同，应该要调整一下半径大小。
- 验证数据也是随机取面片，由于网络加载不了整个口扫，我也是随机取

调整实验：

- downsample面片个数到10000。之前的效果不好可能是因为邻接矩阵的问题，所以这次尝试降采样
- 这次使用牙齿优先采样方法
- 不使用与训练模型，增加MPCNet使用的面片密度信息