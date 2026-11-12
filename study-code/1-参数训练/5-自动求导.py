import torch

# 这次来完成，损失导数的pytorch自动计算

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
# torch.tensor(列表, dtype=类型)
#   作用：把 Python 的列表变成 PyTorch 张量（能参与矩阵运算的多维数组）
#   dtype=torch.float64  每个数字用 64 位存储，双精度，约 15~16 位有效数字
#                        不写 dtype 的话默认 float32（32 位，约 7 位有效数字）
#   这里用 float64 是因为做差分/对照时 float32 精度不够（A03 的 eps 坑）

target = torch.tensor([[2.0], [4.0]], dtype=torch.float64)
# 同上：标准答案。形状 [2,1]，必须和 y 完全一样，否则广播会悄悄算错
# 没写 requires_grad，所以 version 它是"数据"，最后不会算它的梯度

# 只多了 requires_grad=True 这一个参数
w = torch.tensor([[0.3], [0.5]], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
# requires_grad=True   意思："这个张量是要学习的参数，请开始记账"
#   从这一行起，所有用到 w / b 的计算都会被 PyTorch 记进计算图
#   证据就是后面那些 .grad_fn（每个张量都记着"我是由什么运算产生的"）

y = x @ w + b
error = y - target
squared = error ** 2
loss = squared.mean()
# .mean()   把所有元素求平均，结果变成 0 维标量（形状 []，ndim 0）
# 必须是"一个数"才能直接 .backward()；多个数要额外给 gradient 参数

print("loss      =", loss.item())
# .item()   把 0 维张量里的数取出来，变成普通 Python float
#   tensor(0.68)  ->  0.6799999999999999
# 注意：.item() 之后的数就脱离计算图了，不能再拿它去 backward

print("w.grad    =", w.grad)
# .grad   是"属性"，不是函数，所以后面不带括号
#   作用：存放"损失对这个参数的梯度"
#   现在还没 backward 过，所以是 None（不是 0，是"还没算"）

print("w         =", w.tolist())
# .tolist()   把张量转成 Python 的嵌套列表，方便打印和肉眼查看
#   tensor([[0.3],[0.5]])  ->  [[0.3], [0.5]]

print("y.grad_fn =", y.grad_fn)
# .grad_fn  也是"属性"，记录"这个张量是由哪个运算产生的"
# 输出长这样：<AddBackward0 object at 0x0000017988097C10>
#   Add       运算类型：加法
#   Backward  表示它知道怎么反向求导
#   0         版本序号，不用管（不同 PyTorch 版本里可能变化）
#   0x...     内存地址，每次运行都会变，没有任何含义，不用记

loss.backward()          # 就这一行，梯度和手算的一样
# .backward()   沿计算图从 loss 倒着走回去，按链式法则累乘每段的局部变化率
#   走到 w 时把结果存进 w.grad，走到 b 时存进 b.grad
# 关键：它只算梯度，不改参数！backward 之后 w 的值一点没变

print("\n\n--- backward 之后 ---\n\n")
print("w.grad =", w.grad)
# 现在有值了：tensor([[-3.6],[-5.2]], dtype=torch.float64)
# 形状和 w 完全一样 [2,1] —— 梯度形状必须等于参数形状，这条以后用来排查错误

print("b.grad =", b.grad)
# tensor([-1.6], dtype=torch.float64)   形状 [1]，和 b 一样

print("w      =", w.tolist())
# 还是 [[0.3], [0.5]]，证明 backward 只算坡度，没有推门（改参数）