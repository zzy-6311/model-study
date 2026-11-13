import torch
import torch.nn as nn

# B07：多层网络
#
# 一个层太少了。这一课搭 2 -> 4 -> 1 的两层网络，讲两种写法：
#   写法一：显式  —— 一层层写清楚，适合结构复杂、有分支的网络
#   写法二：Sequential —— 排成一队，适合一路直着往下走的网络
# 两种写法算的是同一件事，最后要对拍（输出完全相等）


# ============================================================
# 零、API 说明书
# ============================================================
# nn.Sequential(a, b, c)     把几个层串成一队，数据依次通过
#                            （本身也是个 nn.Module，可以当成员挂上去）
# model.net[0]               取 Sequential 里的第 0 个元素
# nn.ModuleList([a, b])      把几个层装进一个"列表"，并登记进模型
#                            注意：它只负责注册，不会自动执行
# for layer in module_list   可以像列表一样遍历
# named_children()           (名字, 子层) 迭代器，用来逐层观察

torch.manual_seed(0)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])     # 2 个样本，每个 2 个特征


# ============================================================
# 一、写法一：显式
# ============================================================
print("=" * 62)
print("一、显式写法：2 -> 4 -> 1")


class ExplicitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 4)      # 进 2 个特征，出 4 个
        self.fc2 = nn.Linear(4, 1)      # 进 4 个，出 1 个

    def forward(self, x):
        h = self.fc1(x)                 # 第一层
        return self.fc2(h)              # 第二层


m1 = ExplicitNet()
print("参数清单：")
for name, p in m1.named_parameters():
    print(f"   {name:12} shape={tuple(p.shape)}")
# fc1.weight  shape=(4, 2)     4 个输出 × 2 个输入
# fc1.bias    shape=(4,)
# fc2.weight  shape=(1, 4)
# fc2.bias    shape=(1,)

print("总参数量 =", sum(p.numel() for p in m1.parameters()))
# 17 = (4*2 + 4) + (1*4 + 1) = 12 + 5

print("逐层形状：")
h = x
for name, layer in m1.named_children():
    h = layer(h)
    print(f"   过 {name:4} 后 = {tuple(h.shape)}")
# 过 fc1  后 = (2, 4)
# 过 fc2  后 = (2, 1)
# 输入 [2,2]，两个样本、每个两个特征；输出 [2,1]，两个样本、每个一个数


# ============================================================
# 二、写法二：Sequential
# ============================================================
print("=" * 62)
print("二、Sequential 写法")


class SeqNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Linear(4, 1),
        )

    def forward(self, x):
        return self.net(x)              # 一条队，直接传进去


m2 = SeqNet()
print("参数清单：")
for name, p in m2.named_parameters():
    print(f"   {name:14} shape={tuple(p.shape)}")
# net.0.weight  shape=(4, 2)     名字变成"第 0 个"
# net.0.bias    (4,)
# net.1.weight  (1, 4)
# net.1.bias    (1,)

print("总参数量 =", sum(p.numel() for p in m2.parameters()))    # 17，和显式版一样


# ============================================================
# 三、对拍：两种写法是不是同一个东西
# ============================================================
print("=" * 62)
print("三、对拍")

with torch.no_grad():
    m2.net[0].weight.copy_(m1.fc1.weight)      # net[0] 就是第一层
    m2.net[0].bias.copy_(m1.fc1.bias)
    m2.net[1].weight.copy_(m1.fc2.weight)
    m2.net[1].bias.copy_(m1.fc2.bias)

y1 = m1(x)
y2 = m2(x)
print("显式版输出 =", y1.tolist())
print("Seq  版输出 =", y2.tolist())
print("完全相等?", torch.equal(y1, y2))
# True —— 两种写法只是"组织方式"不同，算的是同一个函数


# ============================================================
# 四、ModuleList：注册了，但不会自己执行
# ============================================================
print("=" * 62)
print("四、普通 list 和 ModuleList")


class PlainListNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(2, 4), nn.Linear(4, 1)]     # 普通 Python 列表

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ModuleListNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 4), nn.Linear(4, 1)])   # 会注册

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


p1 = PlainListNet()
p2 = ModuleListNet()
print("普通 list 的参数量   =", sum(p.numel() for p in p1.parameters()))   # 0
print("ModuleList 的参数量  =", sum(p.numel() for p in p2.parameters()))   # 17
print("ModuleList 参数名 =", [n for n, _ in p2.named_parameters()])
# ['layers.0.weight', 'layers.0.bias', 'layers.1.weight', 'layers.1.bias']

# 两件事要分清：
#   注册   ：放进 ModuleList 才会被登记（普通 list 不会，参数量是 0）
#   执行   ：ModuleList 不会自动执行，forward 里必须自己写 for 循环
#
# 对比 Sequential：
#   Sequential  = 注册 + 自动按顺序执行
#   ModuleList  = 只注册，执行逻辑自己写


# ============================================================
# 五、什么时候用哪个
# ============================================================
# Sequential   一条直线往下走：x -> A -> B -> C -> 输出
#              代码最短，适合搭顺序堆叠的结构
#
# ModuleList   需要自己控制流程：循环、条件、跳跃连接
#              比如 "重复同一个块 N 次"，只能自己写 for
#
# 显式成员     结构里有分支、有跨层引用（比如残差 x + f(x)）
#              直接用 self.a = ... 最清楚


# ============================================================
# 六、注意：现在还没有激活函数
# ============================================================
# 2 -> 4 -> 1 这两层，中间没有加任何非线性
# 数学上，"线性层套线性层"仍然等价于"一个线性层"
#   y = W2(W1x + b1) + b2 = (W2W1)x + (W2b1 + b2)
# 所以这个网络的实际表达能力，和一个 Linear(2,1) 一样
#
# 下一课 B08 讲激活函数，加上 ReLU 之后它才真的变成"多层"


# ============================================================
# 总结
# ============================================================
# nn.Sequential      串成一队，注册 + 自动执行，适合直线结构
# nn.ModuleList      只注册不执行，适合需要自己写循环的结构
# 普通 list          不注册，参数量为 0，等于白写
# 逐层看 shape       用 named_children() 遍历，一层层打印
# 对拍              结构怎么写不重要，参数灌成一样，输出必须相同
