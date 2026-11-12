import torch

# A 阶段最后一课：为什么 loss 能降到很小，参数却不是你以为的那一组？
#   核心答案：参数能被唯一确定，前提是"数据里有足够多互相独立的信息"
#   A10 要证明三件事：
#     1. 两条样本解不出三个参数 -> 无穷多组解，loss 都能到 0
#     2. 给三个"各管一个参数"的独立点 -> 参数被精确钉死
#     3. 数据带噪声时，训练误差有下限，永远降不到 0


# ============================================================
# 一、旧例子的真相：三组完全不同的参数，loss 全是 0
# ============================================================
print("=" * 62)
print("一、两条样本 + 三个参数 = 无穷多组解")

x_old = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
t_old = torch.tensor([[2.0], [4.0]], dtype=torch.float64)

for k in (0.1, 0.35, 0.6):
    w = torch.tensor([[k], [1 - k]], dtype=torch.float64)   # w1 = k, w2 = 1-k
    b = torch.tensor([k], dtype=torch.float64)              # b = k
    y = x_old @ w + b
    loss = ((y - t_old) ** 2).mean()
    print(f"k={k:.2f}  w=[{k:.2f}, {1-k:.2f}]  b=[{k:.2f}]  "
          f"y={y.flatten().tolist()}  loss={loss.item():.1e}")

# 三组 (w1, w2, b) 完全不同，预测却一模一样，loss 都是 0
# 因为 (k, 1-k, k) 这一族参数都满足那两条方程：
#   方程1: w1 + 2*w2 + b = 2      方程2: 3*w1 + 4*w2 + b = 4
# 两个方程、三个未知数 -> 解是一条线，不是一个点


# ============================================================
# 二、三个"各管一个参数"的点
# ============================================================

# 这里是我们模拟一个函数y=2 * x1 - 3 * x2 + 0.5，然后用这函数上的3个真实点来计算，训练我们的模拟
# t3是使用我们的函数通过我们的素材点算出来的目标

print("=" * 62)
print("二、三个独立点，各自约束一个参数")

x3 = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
# 点1 [0,0]:  y = w1*0 + w2*0 + b = b        -> 只含 b
# 点2 [1,0]:  y = w1*1 + w2*0 + b = w1 + b   -> 含 w1
# 点3 [0,1]:  y = w1*0 + w2*1 + b = w2 + b   -> 含 w2
# 三条式子恰好一个管一个参数，信息"独立"，所以能全解出来

x1 = x3[:, 0:1]      # 第 0 列（所有行的第一个数），形状 [3,1]
x2 = x3[:, 1:2]      # 第 1 列（所有行的第二个数），形状 [3,1]
# x3[:, a:b]   逗号前选行、逗号后选列；单个 : 表示"全部"
#   0:1  从第 0 列开始、到第 1 列之前（不含 1）-> 只有第 0 列
#   结果是 [[0],[1],[0]]，二维 [3,1]
# 如果写成 x3[:, 0]（不带冒号），这一维会被压掉，变成一维 [3]
#   后面 (y - t) 就会广播成 [3,3] 悄悄算错 —— 就是 A02 讲过的那个坑

print("x1 =", x1.flatten().tolist())     # [0.0, 1.0, 0.0]
print("x2 =", x2.flatten().tolist())     # [0.0, 0.0, 1.0]

t3 = 2 * x1 - 3 * x2 + 0.5
# 真实规律：y = 2*x1 - 3*x2 + 0.5
# 三个列向量逐行各算一次，一次得到 3 个标签
print("t3 =", t3.flatten().tolist())     # [0.5, 2.5, -2.5]

w = torch.tensor([[0.0], [0.0]], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
# ★ 从 0 出发（不是 0.3），证明不是靠初始值猜中的

lr = 0.1
for step in range(4001):
    w.grad = None
    b.grad = None
    y = x3 @ w + b
    loss = ((y - t3) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

print("学到的 w =", [round(v, 4) for v in w.flatten().tolist()],
      " b =", round(b.item(), 4))
print("真实   w = [2.0, -3.0]  b = 0.5")
print("loss =", f"{loss.item():.2e}")
# 3 个点、3 个未知数 -> 唯一解，参数被精确恢复


# ============================================================
# 三、换成 20 个随机点（无噪声）
# ============================================================
print("=" * 62)
print("三、20 个随机点，无噪声")

torch.manual_seed(1)     # 固定随机种子，保证每次跑出来一样
N = 20
x_rand = torch.rand(N, 2, dtype=torch.float64) * 2 - 1
# torch.rand(N, 2)  生成 [N,2] 的随机数，范围 [0,1)
# * 2 - 1           映射到 [-1, 1)
t_rand = 2 * x_rand[:, 0:1] - 3 * x_rand[:, 1:2] + 0.5

w = torch.tensor([[0.0], [0.0]], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

for step in range(2001):
    w.grad = None
    b.grad = None
    y = x_rand @ w + b
    loss = ((y - t_rand) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

print("学到的 w =", [round(v, 4) for v in w.flatten().tolist()],
      " b =", round(b.item(), 4))
print("loss =", f"{loss.item():.2e}")
# 点更多不会让答案"更对"，只是让参数被约束得更紧、拟合更快


# ============================================================
# 四、标签带噪声：训练误差有下限
# ============================================================
print("=" * 62)
print("四、标签加了 0.2 的噪声")

torch.manual_seed(2)
t_noisy = t_rand + 0.2 * torch.randn(N, 1, dtype=torch.float64)
# torch.randn  标准正态随机数（均值 0、标准差 1），* 0.2 就是标准差 0.2
# 现实数据一定有这种"测不准"的成分，不可能完全按公式来

w = torch.tensor([[0.0], [0.0]], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)

for step in range(2001):
    w.grad = None
    b.grad = None
    y = x_rand @ w + b
    loss = ((y - t_noisy) ** 2).mean()
    loss.backward()
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

print("学到的 w =", [round(v, 4) for v in w.flatten().tolist()],
      " b =", round(b.item(), 4))
print("训练 loss =", round(loss.item(), 5))
# 参数不再精确等于 [2, -3, 0.5]，而是有偏差
# 训练 loss 也降不到 0：噪声本身就是随机的，任何模型都拟合不了它


# ============================================================
# 五、留出验证集：训练误差 vs 泛化误差
# ============================================================
print("=" * 62)
print("五、拿没见过的数据检查")

torch.manual_seed(99)
x_val = torch.rand(10, 2, dtype=torch.float64) * 2 - 1
t_val_clean = 2 * x_val[:, 0:1] - 3 * x_val[:, 1:2] + 0.5
# 干净的标签：真正遵循规律的答案
t_val_noisy = t_val_clean + 0.2 * torch.randn(10, 1, dtype=torch.float64)
# 带噪标签：模拟真实标注

pred = x_val @ w + b
print("验证 loss（对比干净答案）=", round((((pred - t_val_clean) ** 2).mean()).item(), 5))
print("验证 loss（对比带噪答案）=", round((((pred - t_val_noisy) ** 2).mean()).item(), 5))

# 关键观察：
#   训练 loss 0.0375，但对比"干净答案"的验证 loss 只有 0.0007
#   说明：模型学的规律是对的，训练 loss 高是因为标签有噪，不是模型差
#   而对比"带噪答案"的验证 loss 约 0.05，正好接近噪声方差 0.04
#   -> 这就是噪声下限，再好的模型也穿不过去

# ============================================================
# 总结
# ============================================================
# 1. 参数唯一确定的前提：数据提供足够多、互相独立的信息
#    2 条样本 < 3 个未知数  ->  无穷多解，loss = 0 也不代表参数对
# 2. 训练误差降不到 0 是正常的：数据本身带噪声
# 3. 训练误差小 != 学得好。要看没见过的数据（验证集）表现
#    这就是"泛化"，C 阶段正式展开
