import torch
import torch.nn as nn

# B09：优化器 —— 把"手动更新参数"交给 torch.optim
#
# 你在 A06/A08 手写过：
#     for p in model.parameters():
#         p.grad = None          # 清梯度
#     ...
#         p -= lr * p.grad       # 更新
#
# 优化器就是把这套动作封装起来，还顺便支持 Momentum、Adam 这些改进


# ============================================================
# 零、API 说明书
# ============================================================
# torch.optim.SGD(参数列表, lr=0.01)        最基础的优化器：按梯度直接走
# torch.optim.SGD(..., momentum=0.9)       带惯性：把历史方向也考虑进去
# torch.optim.Adam(参数列表, lr=0.001)      自适应步长：每个参数用自己的速度
#
# optimizer.zero_grad()     把参数列表里所有参数的 .grad 清成 None（等价 A07 的 p.grad = None）
# optimizer.step()          按梯度更新所有参数（等价 p -= lr * p.grad）
# optimizer.state_dict()    优化器自己的内部状态（动量、历史梯度等），续训时要存
#
# 注意分工：
#   step()  只负责"用已有的 .grad 更新参数"
#   它不算 loss，也不做 backward

torch.manual_seed(0)


# ============================================================
# 一、标准训练循环（四步）
# ============================================================
#   ① optimizer.zero_grad()     清梯度
#   ② 前向 + 算 loss
#   ③ loss.backward()           算梯度
#   ④ optimizer.step()          更新参数
#
# 和 A08 手写版一一对应，只是 ①④ 换成了优化器的两个方法


# ============================================================
# 二、对拍：手写更新 vs optim.SGD
# ============================================================
print("=" * 62)
print("一、对拍：两种更新方式是不是一样")


x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
target = torch.tensor([[2.0], [4.0]])
lr = 0.01


def build():
    torch.manual_seed(1)         # 固定种子，保证两个模型初值完全一样
    return nn.Sequential(
        nn.Linear(2, 3),
        nn.ReLU(),
        nn.Linear(3, 1),
    )


# ① 手写版
m1 = build()
for p in m1.parameters():
    p.grad = None
loss1 = ((m1(x) - target) ** 2).mean()
loss1.backward()
with torch.no_grad():
    for p in m1.parameters():
        p -= lr * p.grad

# ② 优化器版
m2 = build()
opt = torch.optim.SGD(m2.parameters(), lr=lr)     # 传的是参数迭代器
opt.zero_grad()                                    # 等价 p.grad = None
loss2 = ((m2(x) - target) ** 2).mean()
loss2.backward()
opt.step()                                         # 等价 p -= lr * p.grad

diffs = [(a - b).abs().max().item() for a, b in zip(m1.parameters(), m2.parameters())]
print("两种方式更新后的最大参数差 =", max(diffs))
# 0.0  —— 完全一样

# 结论：optim.SGD(lr) 就是"p -= lr * p.grad"的封装，没有别的魔法


# ============================================================
# 三、优化器持有的是"参数对象本身"
# ============================================================
print("=" * 62)
print("二、优化器拿着什么")

m3 = build()
opt3 = torch.optim.SGD(m3.parameters(), lr=0.01)
print("优化器管的参数个数 =", len(opt3.param_groups[0]["params"]))

# ★ 创建时机：循环外面，创建一次
#   因为优化器内部会存"历史信息"（Momentum 的速度、Adam 的均值）
#   如果在循环里每一步都重建优化器，历史每次都被清零，Adam/Momentum 就失效了
#   表现得像普通 SGD，而且不会报错 —— 很难察觉的坑


# ============================================================
# 四、Momentum：给参数加上"惯性"
# ============================================================
print("=" * 62)
print("三、Momentum 多存了什么")

m4 = build()
opt_m = torch.optim.SGD(m4.parameters(), lr=0.01, momentum=0.9)
opt_m.zero_grad()
((m4(x) - target) ** 2).mean().backward()
opt_m.step()

state = opt_m.state_dict()["state"]
print("Momentum 的缓冲（第一组）:")
for k, s in list(state.items())[:2]:
    print("   参数", k, " momentum_buffer =", [round(v, 4) for v in s["momentum_buffer"].flatten().tolist()])

# 普通 SGD 的 state 是空的 {}；加了 momentum 之后，每个参数多存一个"速度"
#
# 直觉：把参数想成一个小球从山坡滚下来
#   普通 SGD      : 每一步只看"当前的坡"往哪边斜
#   Momentum      : 记住上一段的速度，越滚越快；坡换了方向也不会立刻刹住
#   好处：能冲过小坑、在平缓区走得更快
#   公式：v = momentum * v + grad ;  参数 -= lr * v


# ============================================================
# 五、Adam：每个参数用自己的步长
# ============================================================
print("=" * 62)
print("四、SGD 和 Adam 跑同一个任务")

# 任务：拟合 y = x^2（B08 用过）
xs = torch.linspace(-1, 1, 200).reshape(-1, 1)
ys = xs ** 2


def make_net():
    torch.manual_seed(1)
    return nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def run(kind, steps=500):
    torch.manual_seed(1)
    net = make_net()
    if kind == "SGD":
        o = torch.optim.SGD(net.parameters(), lr=0.05)
    else:
        o = torch.optim.Adam(net.parameters(), lr=0.01)

    marks = {1: None, 11: None, 51: None, 101: None, 500: None}
    for i in range(steps):
        o.zero_grad()
        loss = ((net(xs) - ys) ** 2).mean()
        loss.backward()
        o.step()
        if (i + 1) in marks:
            marks[i + 1] = round(loss.item(), 5)
    return marks


print("SGD  lr=0.05 :", run("SGD"))
print("Adam lr=0.01 :", run("Adam"))
# SGD  : {1: 0.25107, 11: 0.07751, 51: 0.06094, 101: 0.04358, 500: 0.00525}
# Adam : {1: 0.25107, 11: 0.08447, 51: 0.01537, 101: 0.00405, 500: 0.00038}
#
# 观察：
#   第 1 步两者完全相同（初值一样、都是按梯度走第一步）
#   后面 Adam 明显更快：500 步时 loss 低了一个数量级
#
# Adam 的直觉：
#   给每个参数单独维护"历史上梯度多大"的记录，
#   梯度大的参数 -> 步长自动调小；梯度小的 -> 步长调大
#   再加上动量，所以在不同尺度的问题上比较省心
#   代价：多存两倍的状态，收敛速度不一定比调好的 SGD 快


# ============================================================
# 六、B 阶段结业：一个完整能训的程序
# ============================================================
print("=" * 62)
print("五、完整例子：模型类 + 优化器 + 训练循环")


class CurveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


torch.manual_seed(0)
model = CurveNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(1001):
    optimizer.zero_grad()                       # ① 清梯度
    pred = model(xs)                            # ② 前向
    loss = ((pred - ys) ** 2).mean()            #    算损失
    loss.backward()                             # ③ 反向
    optimizer.step()                            # ④ 更新

    if step % 200 == 0:
        print(f"step {step:4d}  loss = {loss.item():.6f}")

print("最终 loss =", round(loss.item(), 6))
for px in (-1.0, -0.5, 0.0, 0.5, 1.0):
    t = torch.tensor([[px]])
    print(f"   x={px:+.1f}  预测={model(t).item():.3f}  真实={px**2:.3f}")


# ============================================================
# 总结
# ============================================================
# optimizer.zero_grad()   清梯度，等价 p.grad = None
# optimizer.step()        更新参数，等价 p -= lr * p.grad
# 创建时机                循环外创建一次，循环内只调用方法
# SGD                     只看当前梯度
# Momentum                额外记速度，有惯性
# Adam                    额外记梯度的一阶/二阶历史，每个参数自适应步长
# 分工                    step 不算 loss，也不 backward
