import torch

# 现在知道了损失，还需要回答怎么优化消除损失，知道参数是大了还是小了，导数就是回答“往哪边改”的第一步

w = torch.tensor([0.9, 1.0, 1.1], dtype=torch.float64)
L = (w - 3) ** 2 # 这里w模拟的是结果，3是目标，**2是平方，所以l是平方损失

print(w.tolist())
print(L.tolist())

# 计算可以发现，w从0.9~1.1，越大误差越小，所以在1附近应该是w越大越好，这个损失的变化率也就是数学上的导数

# 计算导数
eps = 0.01 # 变换值，也就是x的变化，也不要太小了，会出现无法存储的问题的
w0 = 1.0 # 原值

pair = torch.tensor([w0 - eps, w0 + eps], dtype=torch.float64) # 通过加减来变化因变量
Lp = (pair - 3) ** 2 # 计算两次的误差

slope = (Lp[1] - Lp[0]) / (2 * eps) # 导数
print(Lp)
print(slope)
print(2 * (w0 - 3)) # 数学上的真导数，用来对照

