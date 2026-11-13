import torch

# B03：dtype（数据类型）/ device（放在哪）/ 和 NumPy 的边界
#
# 前两课解决"形状"，这一课解决"类型和位置"
# 三个必答问题：
#   1. 为什么 loss 打印出来带 0.6800000071525574 这种尾巴？
#   2. 模型和数据不在同一个设备上，会怎么报错？
#   3. .detach() 到底断了什么？它和 clone 有什么本质区别？

print("=" * 62)
print("零、先看环境")
print("torch 版本 :", torch.__version__)
print("CUDA 可用  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("显卡型号   :", torch.cuda.get_device_name(0))
# cuda:0 表示第 0 号 GPU


# ============================================================
# 一、四种 dtype，各管什么
# ============================================================
print("=" * 62)
print("一、dtype 四种")

demo = {
    "float32": torch.tensor([1.5], dtype=torch.float32),   # 默认，训练用
    "float64": torch.tensor([1.5], dtype=torch.float64),   # 高精度，做对照
    "int64":   torch.tensor([1, 2], dtype=torch.int64),    # 分类标签
    "bool":    torch.tensor([True, False]),                # 掩码 / 条件
}
for k, v in demo.items():
    print(f"{k:8} shape={tuple(v.shape)} dtype={v.dtype}")

# float32  32 位浮点，约 7 位有效数字   -> 训练默认，省显存、算得快
# float64  64 位浮点，约 15 位有效数字  -> 数值对照、差分实验（A03 讲过 eps 的坑）
# int64    64 位整数                    -> 分类任务的标签必须是这种
# bool     布尔                         -> 用于 mask，比如 [True, False] 挑元素


# ============================================================
# 二、float32 的精度到底损失在哪
# ============================================================
print("=" * 62)
print("二、0.1 存不准")

a32 = torch.tensor(0.1, dtype=torch.float32)
a64 = torch.tensor(0.1, dtype=torch.float64)
print("float32 的 0.1 =", repr(a32.item()))
print("float64 的 0.1 =", repr(a64.item()))
# float32 的 0.1 = 0.10000000149011612
# float64 的 0.1 = 0.1
#
# 原因：0.1 在二进制里是无限循环小数，任何位数都存不下精确值
#       float32 只留 32 位，误差在 1e-8 量级；float64 留 64 位，误差小到看不出来

s32 = torch.zeros((), dtype=torch.float32)
for _ in range(10):
    s32 = s32 + a32
print("float32 把 0.1 加十次 =", repr(s32.item()), "（应该是 1.0）")
# 得到 1.0000001192092896 —— 这就是误差累积

# ★ 两个直接后果：
#   1. 绝不能用 == 比较浮点数，要写成 abs(a - b) < 1e-6
#   2. 训练日志里那些 0.6800000071525574 不是 bug，是 float32 的正常表现
#   3. 所以 A 阶段我们一直用 float64：要做差分、和手算对照，float32 的尾巴会干扰判断


# ============================================================
# 三、类型转换：返回的是新张量
# ============================================================
print("=" * 62)
print("三、转换不改原来的")

x = torch.tensor([1, 2, 3])
print("原张量 dtype =", x.dtype)          # int64
y = x.float()
print("转换后 y   dtype =", y.dtype)      # float32
print("x 有没有变？dtype =", x.dtype)     # 还是 int64
print("y is x ?", y is x)                # False
# .float() / .double() / .long() / .to(torch.xxx) 都是"复制成新张量"
# 想改自己得写 x = x.float()

# 另一种写法，效果一样，更通用：
z = x.to(torch.float64)
print("x.to(float64) ->", z.dtype)


# ============================================================
# 四、device：张量住在哪
# ============================================================
print("=" * 62)
print("四、CPU 和 GPU")

cpu_t = torch.tensor([1.0, 2.0])
print("默认建在 :", cpu_t.device)        # cpu

if torch.cuda.is_available():
    gpu_t = torch.tensor([1.0, 2.0], device="cuda")
    print("指定 cuda :", gpu_t.device)   # cuda:0

    print("搬回 CPU :", gpu_t.cpu().device)

    # 两个张量在不同设备上直接相加 -> 报错
    try:
        gpu_t + cpu_t
    except RuntimeError as e:
        print("混算报错 ->", str(e)[:105])

    # 正确做法：要么都搬到 GPU，要么都搬到 CPU
    print("都搬到 GPU :", (gpu_t + torch.tensor([3.0, 4.0], device="cuda")).cpu().tolist())
else:
    print("这台机器没有可用 CUDA，跳过")

# ★ 铁律：参与同一个运算的张量，必须在同一个设备上
#   后面模型的参数也要和输入在同一设备：model.to("cuda") 和 x.to("cuda")
#   报错关键词：Expected all tensors to be on the same device


# ============================================================
# 五、detach 和 clone：完全不同的两件事
# ============================================================
print("=" * 62)
print("五、detach vs clone")

w = torch.tensor([1.0, 2.0], requires_grad=True)
loss = (w * 3).sum()
print("loss.requires_grad =", loss.requires_grad)     # True

d = loss.detach()
c = loss.clone()
print("detach 后 requires_grad =", d.requires_grad)   # False  图断了
print("clone  后 requires_grad =", c.requires_grad)   # True   图还在
print("detach 和原张量共享内存?", d.data_ptr() == loss.data_ptr())   # True
print("clone  和原张量共享内存?", c.data_ptr() == loss.data_ptr())   # False

# 共享内存意味着：改一个，另一个也变！
v = w.detach()
print("w.detach() 与 w 共享内存?", v.data_ptr() == w.data_ptr())    # True
v[0] = 99.0
print("改了 detach 之后，w =", w.tolist())          # [99.0, 2.0]  <- w 被改了！

# ★ 一句话区分：
#   detach()  断开计算图，但和原张量共用同一块内存（不是拷贝！）
#   clone()   复制一份数据，计算图关系保持不变
#
#   想"既要断开又要独立"，写 .detach().clone()
#   只想"取个数值看看"，用 .item() 最省事


# ============================================================
# 六、和 NumPy 的边界
# ============================================================
print("=" * 62)
print("六、转 NumPy")

t = torch.tensor([1.0, 2.0])
print("普通张量 ->", t.numpy())
# numpy() 和原张量共享内存，改一边另一边也变

try:
    torch.tensor([1.0], requires_grad=True).numpy()
except RuntimeError as e:
    print("带梯度 ->", str(e)[:95])
# Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
# 原因：NumPy 没有"计算图"这个概念，转过去梯度就丢了，所以 PyTorch 直接拦下

if torch.cuda.is_available():
    try:
        torch.tensor([1.0], device="cuda").numpy()
    except TypeError as e:
        print("GPU 张量 ->", str(e)[:95])
# can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() ...

# ★ 所以标准写法是三段式：
#     t.detach().cpu().numpy()
#         detach  断图（解决 requires_grad 报错）
#         cpu     搬回内存（解决 GPU 报错）
#         numpy   转换
#   三段各解决一个问题，少一段就会报错

# 但如果你只是想要"一个数字"，别绕这么远，直接 .item() 就行


# ============================================================
# 总结
# ============================================================
# dtype   训练用 float32；对照/差分用 float64；标签用 int64；mask 用 bool
#         浮点不能 == 比较，误差来自二进制存不下 0.1 这种数
# device  参与同一运算的张量必须在同一设备；报错关键词 "same device"
# detach  断图 + 共享内存（改它会改原张量）
# clone   复制 + 保留图
# numpy   .detach().cpu().numpy() 三段式，缺一段报错
