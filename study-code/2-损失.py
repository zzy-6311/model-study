import torch

# 这次来了解什么是损失，也就是在函数学习中的损失

x = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])
w = torch.tensor([
    [0.3],
    [0.5]
])
b = torch.tensor([0.1])
y = x @ w + b
print(y) # 这是上一章里面学习到了的，通过张量构建一个最简单的线性层

# 但模型不知道自己猜得对不对，所以加上一个标准答案
target = torch.tensor([
    [2.0],
    [4.0]
])
print(target)
error = y - target # 这个就是计算估算的误差，注意这里需要y - target，所以必须要求y和target两个数据需要一样才行，不然有问题
print(error)

squared_error = error ** 2 # 均方误差，避免有正有负
print(squared_error)

loss = squared_error.mean() # 这个就是现在这个模型的误差，.mean()就是均分误差求平均
print(loss)

