import torch
import torch.nn as nn

# B06：参数追踪
#
# 本课只回答一个问题：
#   写 self.xxx = ... 的时候，模型到底认不认这个成员？
#
#   认（登记了）：优化器能找到它、.to("cuda") 会带走它、存盘时会存它
#   不认（没登记）：只是个普通的属性，训练时完全被忽略


# ============================================================
# 零、API 说明书（先看这里，再往下看代码）
# ============================================================
# 【建模】
# nn.Module                     模型基类。继承它，才有"自动登记参数"的能力
# class Net(nn.Module):         定义一个模型类，表示"Net 是一种模型"
# super().__init__()            执行父类初始化，建好参数表。模型类第一行必须有
# nn.Linear(2, 1)               造一个线性层：进 2 个数、出 1 个数。
#                               内部自动建好 weight（形状[1,2]）和 bias（形状[1]）
# self.fc = ...                 把层存成对象的成员（同时被登记）
# def forward(self, x):         定义"输入 x 之后怎么算"。只是定义，不会执行
#
# 【调用】
# model(x)                      真正执行一次前向计算，返回输出张量
#
# 【查看内部】
# print(model)                  打印模型结构，自动列出所有子层
# model.parameters()            迭代器：每次吐出一个参数张量
# model.named_parameters()      迭代器：每次吐出 (名字, 张量)
# model.named_buffers()         迭代器：每次吐出 (名字, buffer张量)
# model.named_children()        迭代器：每次吐出 (名字, 子层)
# model.state_dict()            字典 {名字: 张量}，存盘/加载权重就靠它
# p.numel()                     一个张量里有几个数字
# p.shape / p.dtype             形状 / 数据类型
#
# 【把迭代器变简单】
# list(迭代器)                  装进列表，方便打印
# dict(迭代器)                  把 (名字,张量) 变成 {名字: 张量}
# sum(p.numel() for p in ...)   把每个张量的元素数加起来 = 总参数量
#
# 【成员类型】
# torch.tensor(...)             普通张量：不会登记
# nn.Parameter(张量)            标记成参数：登记 + 自动 requires_grad=True
# model.register_buffer("名字", 张量)  登记成 buffer：跟着模型走，但不训练
#
# 【整体转换】
# model.double()                把登记过的参数和 buffer 全转成 float64
# model.to("cuda")              把登记过的参数和 buffer 全搬到 GPU
#                               注意：普通张量成员不跟着走

torch.manual_seed(0)


# ============================================================
# 一、最小的模型：建 -> 看 -> 算
# ============================================================
print("=" * 62)
print("一、最小模型")


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)


model = TinyNet()
# 三句话读懂这段：
#   class ... (nn.Module)  定义了模型的样子
#   __init__               里面准备零件（这里是 fc 这一层）
#   forward                规定零件怎么接（x 进来先过 fc）

print("模型结构：")
print(model)
# TinyNet(
#   (fc): Linear(in_features=2, out_features=1, bias=True)
# )
# nn.Module 自带的打印，自动列出所有子层

out = model(torch.tensor([[1.0, 2.0]]))
print("输入 [[1,2]] 的输出 =", out.tolist())
print("输出形状 =", tuple(out.shape))       # (1, 1)

# model(x) 是正规调用方式。
# 它内部先调 __call__（做准备工作），再调你写的 forward。
# 直接写 model.forward(x) 能跑，但跳过了准备工作，不要这么用。


# ============================================================
# 二、模型里到底有什么：去看
# ============================================================
print("=" * 62)
print("二、查看模型里的东西")

print("named_parameters():")
for name, param in model.named_parameters():
    # 迭代器每次吐出两样：名字、张量
    print(f"   {name:10} shape={tuple(param.shape)}  numel={param.numel()}")
# fc.weight  shape=(1, 2)  numel=2
# fc.bias    shape=(1,)    numel=1

print("总参数量 =", sum(p.numel() for p in model.parameters()))    # 3

print("state_dict() 的键 =", list(model.state_dict().keys()))
# ['fc.weight', 'fc.bias']   这就是存盘时的字段名

print("named_children() =", [n for n, _ in model.named_children()])
# ['fc']   子层只有一个

# 对比一下"迭代器"和"列表"：
print("parameters() 打印出来 =", model.parameters())
# <generator object ...>  一个迭代器，不能直接看内容
# 所以要用 for 遍历，或者 list(...) / dict(...) 装起来


# ============================================================
# 三、三种成员，区别只在"登不登记"
# ============================================================
print("=" * 62)
print("三、普通张量 / Parameter / buffer")


class DemoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)                          # 子层：自动登记
        self.plain = torch.tensor([2.0])                   # 普通张量：不登记
        self.scaled = nn.Parameter(torch.tensor([2.0]))    # 参数：登记，要训练
        self.register_buffer("my_mean", torch.tensor([1.0]))   # buffer：登记，不训练

    def forward(self, x):
        return self.fc(x) * self.scaled


d = DemoNet()

print("参数 named_parameters =", [(n, tuple(p.shape)) for n, p in d.named_parameters()])
# [('scaled', (1,)), ('fc.weight', (1, 2)), ('fc.bias', (1,))]
# 注意：先自己的参数，再子模块的，不是按代码顺序

print("buffer named_buffers  =", [(n, tuple(b.shape)) for n, b in d.named_buffers()])
# [('my_mean', (1,))]

print("子层 named_children   =", [n for n, _ in d.named_children()])
# ['fc']   Parameter 和 buffer 不算"子层"

print("state_dict 的键       =", list(d.state_dict().keys()))
# ['scaled', 'my_mean', 'fc.weight', 'fc.bias']  参数 + buffer 都在

print("plain 是参数吗？      =", "plain" in dict(d.named_parameters()))   # False

print("总参数量 =", sum(p.numel() for p in d.parameters()))    # 4 = 1 + 2 + 1

# 三种成员一句话：
#   普通张量  只是一条属性，模型不管，训练/搬设备/存盘都没它的份
#   Parameter 登记成参数，要训练，优化器能找到
#   buffer    登记但不算参数，仍会跟着模型搬设备、存盘
#             典型用途：BatchNorm 的 running_mean、位置编码表


# ============================================================
# 四、登记的后果：谁跟着模型走
# ============================================================
print("=" * 62)
print("四、.double() 之后谁变了")

d64 = DemoNet().double()
# .double() 把所有登记过的参数和 buffer 转成 float64

print("fc.weight dtype =", d64.fc.weight.dtype)    # torch.float64  子层
print("scaled    dtype =", d64.scaled.dtype)       # torch.float64  Parameter
print("my_mean   dtype =", d64.my_mean.dtype)      # torch.float64  buffer
print("plain     dtype =", d64.plain.dtype)        # torch.float32  ★ 没跟着走

# 所以 "登记" 不是一个概念，它有具体后果：
#   登记过的：.double() / .to("cuda") / .half() 都会处理
#   没登记的：留在原地，之后参与计算容易报 dtype 或 device 的错


# ============================================================
# 五、坑：层建在 forward 里
# ============================================================
print("=" * 62)
print("五、层建在 forward 里会怎样")


class BadNet(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        fc = nn.Linear(2, 1)      # 每次前向都新建一个随机层
        return fc(x)


bad = BadNet()
print("BadNet 参数个数 =", len(list(bad.parameters())))      # 0

x_in = torch.tensor([[1.0, 2.0]])
print("同一输入算两次 =", round(bad(x_in).item(), 4), round(bad(x_in).item(), 4))
# 两个数完全不同，因为每次都是新随机层

# 后果：
#   1. 参数个数 0 -> 优化器没有参数可训 -> 训练完全无效
#   2. 同一输入两次不同 -> 模型没有"记忆"
# 规矩：层建在 __init__ 里，forward 里只负责调用


# ============================================================
# 六、能训的模型长什么样
# ============================================================
print("=" * 62)
print("六、能训的模型")


class RealNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)


net = RealNet()
print("参数清单：")
for name, p in net.named_parameters():
    print(f"   {name:10} shape={tuple(p.shape)}  requires_grad={p.requires_grad}")

# 这个 net 可以直接交给优化器（B09 的内容）：
#   optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
# 优化器只需要一个参数迭代器，别的它自己管


# ============================================================
# 总结
# ============================================================
# 建模型   __init__ 里 self.xxx = 层；forward 里写数据怎么流
# 调用     model(x)
# 查看     named_parameters / named_buffers / named_children / state_dict
# 登记     nn.Linear、nn.Parameter、register_buffer 会自动登记
#          普通 torch.tensor 不会
# 后果     只有登记过的才：能被训练、能搬设备、能存盘
# 三条禁令 1. 层不建在 forward 里
#          2. 层不写在 class 层级（会被所有实例共享）
#          3. 不用 model.forward(x) 调用
