import torch

# 首先第一点就是照片是怎么在pytorch里面表示的，照片的表示方法是张量

image = torch.zeros(1, 3, 4, 4)

# image = torch.zeros(1, 3, 4, 4)，这就表示一张照片，其中1就是照片数量是1，3就是表示照片的通道数量，比如RGB就是3个颜色通道
# 然后两个4表示尺寸为4*4，所以这个就是表达1张具有RGB三个通道、尺寸为4×4的图片（也不一定是RGB，但是数据格式就是三通道）

#张量的参数
print(image.shape)
#张量的数据类型
print(image.dtype)
#张量的存储位置
print(image.device)
# 这三个的输出是：
# torch.Size([1, 3, 4, 4])//告诉你这个张量的形状是(1, 3, 4, 4)，这个张量储存的数字是1*3*4*4，一共是48个数字
# torch.float32//表示储存这个张量的48个数字，他们都是32位浮点
# cpu//表示存储这个数据的位置，因为是随手创建的，所以是在cpu中

# 取出第 1 张图片
print(image[0].shape)
# 取出第 1 张图片的第 1 个通道，我们约定它是红色通道
print(image[0, 0].shape)
# 取出第 1 张图片的第 1 个通道第 1 行、第 1 列的数值
print(image[0, 0, 0, 0])
# 输出是：（张量就是一个多维数组）
# torch.Size([3, 4, 4])//取出的第 1 张图片
# torch.Size([4, 4])//取出的第一张照片的第一个通道
# tensor(0.)//去除的第一张照片第一个通道的第一行第一列
# 这里都是使用的.shape也就是输出的是这个张量这个通道的形状，print(image[0])就可以直接打印第一张照片

# 对张量的指定位置进行修改
image[0, 0, 0, 0] = 1.0
print(image[0, 0])

# 数值直接传入构建一个张量
x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])
print(x) # 打印张量
print(x.shape) # 打印张量的形状参数
print(x + 10) # 对张量的所有值做加法
print(x * 2) # 对张量的所有值做乘法
print(x.mean()) # 对张量求平均

a = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]])
b = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0]])

# 张量还主要是还要用于矩阵运算
print(a * b) # 矩阵的每一位相互相乘
print(a @ b) # 矩阵之间的乘法（线代）

# a = torch.ones(2, 3) # 这个的功能是创建2*3的矩阵，并且参数全部写1
# b = torch.ones(3, 4)
# result = a @ b
# print(result)
# print(result.shape)

print("\n\n\n")

# 了解张量的计算后我们来使用张量计算来完成一个神经网络里面的最简单线性层，也就是参数X权重+偏移（y = x @ w + b）
# 两个样本，每个样本有两个输入特征
x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])
# 两个输入特征，各自对应一个权重
w = torch.tensor([
    [0.3],
    [0.5]
])
# 偏置：在加权求和之后，再加上的一个数
b = torch.tensor([0.1])
# 计算预测值
y = x @ w + b
print(y)
print(y.shape)

