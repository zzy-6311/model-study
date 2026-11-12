import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
target = torch.tensor([[2.0], [4.0]], dtype=torch.float64)

def loss_of(w, b):
    return ((x @ w + b - target) ** 2).mean()

w = torch.tensor([[0.3], [0.5]], dtype=torch.float64)
b = torch.tensor([0.1], dtype=torch.float64)
print("loss =", round(loss_of(w, b).item(), 6))

eps = 1e-4

w_hi = w.clone(); w_hi[0, 0] += eps
w_lo = w.clone(); w_lo[0, 0] -= eps
print("dw1 =", round(((loss_of(w_hi, b) - loss_of(w_lo, b)) / (2 * eps)).item(), 4))

w_hi = w.clone(); w_hi[1, 0] += eps
w_lo = w.clone(); w_lo[1, 0] -= eps
print("dw2 =", round(((loss_of(w_hi, b) - loss_of(w_lo, b)) / (2 * eps)).item(), 4))

b_hi = b.clone(); b_hi[0] += eps
b_lo = b.clone(); b_lo[0] -= eps
print("db  =", round(((loss_of(w, b_hi) - loss_of(w, b_lo)) / (2 * eps)).item(), 4))

# 上面的代码就是通过分别求偏导，因为w有两个未知数，b有一个未知数，如果一起变化没法求，只能通过固定两个求其中一个的方式来计算
# 也就是数学意义上面的偏导，通过偏导就能得到一整个变化梯度，也就是整个参数变化的趋势：
# dw1 = -3.6   负 → 往大调，每调 1，损失降 3.6
# dw2 = -5.2   负 → 往大调，每调 1，损失降 5.2
# db  = -1.6   负 → 往大调，每调 1，损失降 1.6
# 这也就是模型训练中的参数变化，但是我们这样手算是不行的，就需要借助pytorch里面的函数，也就是通过函数来批量的计算偏导和梯度来训练
# 一个模型有上百万参数，我们现在的这个来写就太多了，用pytorch函数就几行，下一章我们就来学习这个