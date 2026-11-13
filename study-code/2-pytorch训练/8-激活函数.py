import torch
import torch.nn as nn

# B08：激活函数
#
# 上一课搭了 2->4->1 的两层网络，但没加激活函数。
# 这一课要证明两件事：
#   1. 不加激活，叠多少层都等于一层（数学上能合并）
#   2. 加了 ReLU，网络才有能力拟合"弯"的关系


# ============================================================
# 零、API 说明书
# ============================================================
# nn.ReLU()                 激活函数：每个数做 max(0, x)
#                           负数变 0，正数不变。没有参数，不改形状
# nn.Sigmoid()              每个数压到 (0, 1)
# nn.Tanh()                 每个数压到 (-1, 1)
# nn.Sequential(a, b, ...)  把层串成一队依次执行（B07 学过）
#
# 注意：激活函数是"无参数层"，parameters() 里看不到它，但它照样是 nn.Module

# ============================================================
# 一、ReLU 到底做了什么
# ============================================================
print("=" * 62)
print("一、ReLU 的输入输出")

relu = nn.ReLU()
t = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
print("输入 =", t.tolist())
print("输出 =", relu(t).tolist())
# [-3,-1,0,1,3] -> [0,0,0,1,3]
# 规则一句话：max(0, x)。负数一律压成 0，正数原样通过

print("形状变吗:", tuple(t.shape), "->", tuple(relu(t).shape))    # 不变
print("有参数吗:", sum(p.numel() for p in relu.parameters()))      # 0


# ============================================================
# 二、为什么必须有激活：两层线性 = 一层线性
# ============================================================
print("=" * 62)
print("二、两层线性可以合并成一层")

torch.manual_seed(0)
f1 = nn.Linear(2, 4)
f2 = nn.Linear(4, 1)

x2 = torch.randn(5, 2)
y_two = f2(f1(x2))          # 两层叠加

# 数学上： y = W2(W1x + b1) + b2 = (W2W1)x + (W2b1 + b2)
W_merged = f2.weight @ f1.weight
b_merged = f2.bias + f2.weight @ f1.bias

single = nn.Linear(2, 1)
with torch.no_grad():
    single.weight.copy_(W_merged)
    single.bias.copy_(b_merged)

y_one = single(x2)
print("两层输出 =", [round(v, 4) for v in y_two.flatten().tolist()])
print("一层输出 =", [round(v, 4) for v in y_one.flatten().tolist()])
print("完全相等?", torch.allclose(y_two, y_one, atol=1e-6))
# True

# 结论：不加激活，"多层"只是写法上的多层，实际表达能力等于一层
#       你上一次得到的 [W2W1, W2b1+b2] 就是那个等效的单层参数


# ============================================================
# 三、加了 ReLU 之后，能拟合弯的曲线
# ============================================================
print("=" * 62)
print("三、拟合 y = x^2")

# 造数据：x 取 [-1, 1] 之间 200 个点，y = x^2（一条开口向上的抛物线）
x = torch.linspace(-1, 1, 200).reshape(-1, 1)
y = x ** 2
# torch.linspace(-1, 1, 200)  在 -1 到 1 之间等距生成 200 个数
# .reshape(-1, 1)             变成 [200, 1]，即 200 个样本、每个 1 个特征


class NetNoAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.Linear(16, 1),        # 中间没有激活
        )

    def forward(self, x):
        return self.net(x)


class NetReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),               # 关键就差这一行
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def train(model, steps=2000, lr=0.05):
    for _ in range(steps):
        for p in model.parameters():
            p.grad = None                  # 清梯度（A07）
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                p -= lr * p.grad           # 手动更新每个参数（B09 会用优化器代替）
    return loss.item()


torch.manual_seed(1)
net_a = NetNoAct()
print("无激活  最终 loss =", round(train(net_a), 6))

torch.manual_seed(1)
net_b = NetReLU()
print("加 ReLU 最终 loss =", round(train(net_b), 6))

# 无激活   最终 loss = 0.090678
# 加 ReLU  最终 loss = 0.001174
#
# 差了 77 倍。因为无激活的网络只能画出"直线"，
# 而 y = x^2 是抛物线，直线怎么调都只能取一个折中位置（0.0907 就是那条最佳直线）
# 加了 ReLU 之后，网络能拼出折线，就贴上了抛物线

# 顺便：两条网络参数量一样（都是 16*1+16 + 1*16+1 = 49），
#       差别只在中间那个 0 参数的 ReLU


# ============================================================
# 四、输出层要不要加激活？看任务
# ============================================================
print("=" * 62)
print("四、输出层的处理")

# ① 回归任务（预测一个数，比如价格、温度、坐标）
#    输出层不加激活，让它能输出任意实数
#    如果加了 ReLU，预测值就永远 >= 0，房价不可能为负还好，
#    但坐标、温度、涨跌幅这些有负值的任务就被卡死了

# ② 分类任务（输出每类的分数）
#    输出层通常也不加，直接给"原始分数"（logits）
#    因为 CrossEntropyLoss 内部已经包含了 softmax，重复加会算错
#    只有在推理、要给"概率"看的时候，才手动 softmax 一下

demo = torch.tensor([[-1.0, 0.0, 2.0]])
print("线性层输出（可正可负）=", nn.Linear(3, 1, bias=False)(demo).tolist())
# 回归要的就是这种可正可负的数

print("如果后面硬加 ReLU  =", nn.ReLU()(torch.tensor([[-1.0, 0.0, 2.0]])).tolist())
# [-1,0,2] -> [0,0,2]，负的全被砍掉了


# ============================================================
# 五、为什么 ReLU 管用（直觉）
# ============================================================
# 一层线性做的事：输入 × 权重 求和 -> 输出是一个"平面/直线"
# 叠加几层，还是平面/直线（第二节证明了）
#
# ReLU 做的事：把负数砍成 0，相当于在 0 处"折了一下"
#   于是输出变成"折线"，多个折线拼起来就能逼近任何曲线
#   折得越多（隐藏单元越多），能拼出的形状越复杂
#
# 这就是"非线性"的意义：不是让结果更复杂，而是让模型能表达"弯曲"


# ============================================================
# 总结
# ============================================================
# nn.ReLU()          逐元素 max(0,x)，无参数、不改形状
# 为什么需要          不加激活，多层线性数学上等于一层
# 放哪里              隐藏层之间（Linear -> ReLU -> Linear ...）
# 输出层             回归不加；分类不加（logits 交给损失函数）
# 注意               训练不好不一定是层数/激活的问题，也可能来自数据、学习率
