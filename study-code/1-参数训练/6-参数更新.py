import torch

# 本章学习训练的参数更新

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
target = torch.tensor([[2.0], [4.0]], dtype=torch.float64)

w = torch.tensor([[0.3], [0.5]], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)

y = x @ w + b
loss = ((y - target) ** 2).mean()

# 还是先照旧创造原来的计算数据

print("更新前 w    =", w.tolist())
print("更新前 b    =", b.tolist())
print("更新前 y    =", y.tolist())
print("更新前 loss =", round(loss.item(), 6))

loss.backward()
print("梯度 w      =", w.grad.tolist())
print("梯度 b      =", b.grad.tolist())

print("\n")

lr = 0.01 # 学习率，也就是每次学习按照坡度走多少
with torch.no_grad(): # 前面的w开启了pytorch的跟踪记录，如果直接无训练修改会报错，这里先关闭（with就是这个开启后才运行缩进的语法）
    w -= lr * w.grad # w和b进行步进修改
    b -= lr * b.grad

print("更新后 w    =", w.tolist())
print("更新后 b    =", b.tolist())
print("旧 y   还是 =", y.tolist())
print("旧 loss     =", round(loss.item(), 6))

y2 = x @ w + b # 重新计算
loss2 = ((y2 - target) ** 2).mean()
print("新 y        =", y2.tolist())
print("新 loss     =", round(loss2.item(), 6)) # 发现损失减小