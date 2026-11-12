import torch

# 这一课要弄清梯度默认是"累加"的，训练时每步必须手动清掉，否则参数会被越推越飞

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
target = torch.tensor([[2.0], [4.0]], dtype=torch.float64)

w = torch.tensor([[0.3], [0.5]], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)
# 照旧创建

def show(t):
    # 这里的作用：把张量打印成好看的一行，比如 [-3.6, -5.2]
    return [round(v, 3) for v in t.flatten().tolist()]
    # .flatten()   把 [2,1] 压平成 [2]，只为打印
    # .tolist()    张量 -> Python 列表
    # round(v, 3)  每个数保留 3 位小数，去掉 0.33599999999999997 这种浮点尾巴
    #              保留 3 位是为了让 0.336 这种小变化也能看清楚
    # [ ... for v in ... ]  列表推导式：把列表里每个元素都处理一遍，组成新列表

print("=" * 55)
print("一、不清梯度，连续三次 前向 + backward")

for i in range(3): # 这里就是默认计算倒算梯度
    y = x @ w + b
    loss = ((y - target) ** 2).mean()
    loss.backward()
    # .backward()  沿计算图反向传播
    #   ★关键行为：梯度是"加到" .grad 上，不是"覆盖"
    #   第一次 .grad 是 None，PyTorch 先当 0 再加 -> -3.6
    #   第二次已经存着 -3.6，再加 -3.6 -> -7.2

    print("第", i + 1, "次  loss =", round(loss.item(), 6),
          " w.grad =", show(w.grad))

# 注意：w 一直是 0.3/0.5，loss 一直是 0.68，只有梯度在涨

# ============================================================
# 二、清梯度的两种写法
# ============================================================
print("=" * 55)
print("二、清梯度")

w.grad = None
b.grad = None
# 写法一：直接设成 None（推荐）
#   作用：把旧的梯度整个扔掉，下一次 backward 从 0 重新开始
#   好处：连内存一起释放
print("设成 None 之后 w.grad =", w.grad)        # 打印出 None

y = x @ w + b
loss = ((y - target) ** 2).mean()
loss.backward()
print("重新算一次后   w.grad =", show(w.grad))   # [-3.6, -5.2]，又是干净的

w.grad.zero_()
print("zero_() 之后   w.grad =", show(w.grad))   # [0.0, 0.0]
# 写法二：.zero_() 原地清零
#   方法名末尾的下划线 _ 表示"原地修改"，改的是这个张量自己，不新建
#   类似的还有 .add_() .mul_() .sub_()
#   区别：zero_() 把数改成 0，内存还占着；= None 直接丢掉，内存释放

# ============================================================
# 三、坑：同一个 loss 不能连续 backward 两次
# ============================================================
print("=" * 55)
print("三、同一个 loss 连续 backward 两次")

w.grad = None
b.grad = None
y = x @ w + b
loss = ((y - target) ** 2).mean()
loss.backward()          # 第一次：正常

try:
    # try: ... except: ...   捕获错误，让程序不崩，继续往下跑
    #   用它只是为了把报错信息打印出来看，属于教学写法
    loss.backward()      # 第二次：会报错
except RuntimeError as e:
    print("报错 ->", str(e)[:60], "...")

# 报错原因：backward 之后，计算图里的中间结果就被释放了（省内存）
# 所以"一次前向只配一次反向"；想再算必须重新前向，造一张新图
# 网上解法 loss.backward(retain_graph=True) 能绕过去，但普通训练不需要
#   它会阻止释放，白占内存

# ============================================================
# 四、正确的一步：清 -> 前向 -> 反向 -> 更新
# ============================================================
print("=" * 55)
print("四、正确顺序走一步")

w.grad = None
b.grad = None
lr = 0.01 # 学习率

y = x @ w + b
loss = ((y - target) ** 2).mean()
loss.backward()

with torch.no_grad():
    # with torch.no_grad():  临时关掉梯度记账
    #   因为 w 是要学习的叶子张量，不允许在记账状态下被原地修改
    w -= lr * w.grad
    b -= lr * b.grad
    # 原地修改（改 w 自己）；如果写成 w = w - ...，会造出新张量，

y2 = x @ w + b
loss2 = ((y2 - target) ** 2).mean()
# 必须重新前向！旧的 y 和 loss 是上一次算出来的死数，不会自动跟着变

print("更新前 loss =", round(loss.item(), 6))
print("更新后 loss =", round(loss2.item(), 6))
print("新 w =", show(w), " 新 b =", show(b))

# 预期：0.68 -> 0.32168