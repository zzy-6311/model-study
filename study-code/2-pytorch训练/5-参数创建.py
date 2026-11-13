import torch
import torch.nn as nn

# B05：用 nn.Linear 复现你手写的线性层
#
# 你手写的东西（A08）：
#     w = torch.tensor([[0.3], [0.5]])       形状 [2,1]
#     b = torch.tensor([0.1])
#     y = x @ w + b
#
# 换成库：
#     layer = nn.Linear(2, 1)
#     y = layer(x)
#
# 这一课只有一个目标：让这两个 y 完全相等
# 对不上，说明还有没弄明白的地方


# ============================================================
# 一、nn.Linear(2, 1) 的两个数字是什么意思
# ============================================================
torch.manual_seed(0)          # 固定随机种子，让下面打印的初始值每次都一样

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])     # [2个样本, 2个特征]

layer = nn.Linear(2, 1)
# 第一个 2 = in_features   每个样本有几个输入特征
# 第二个 1 = out_features  每个样本输出几个数
#
# 注意：这两个数字说的是"特征数"，不是"样本数"
#   样本数 2 是 x 的事，跟层无关
#   层只关心：进来必须是 2 个特征，出去给 1 个数

print("=" * 62)
print("一、层的内部结构")
print("weight shape =", tuple(layer.weight.shape))    # (1, 2)
print("bias   shape =", tuple(layer.bias.shape))      # (1,)
print("weight 需要梯度吗 =", layer.weight.requires_grad)   # True，自动开启
print("weight 初始值 =", layer.weight.tolist())
print("bias   初始值 =", layer.bias.tolist())


# ============================================================
# 二、为什么 weight 是 [1, 2] 而不是 [2, 1]
# ============================================================
# 你手写时 w 是 [2,1]：2 行（每个输入特征一个权重）、1 列（一个输出）
#
# nn.Linear 存的是转置版：weight 是 [out_features, in_features] = [1, 2]
#   [[w1, w2]]   -> 一个输出，需要两个权重
#
# 它内部算的是：
#     y = x @ weight.T + bias
#              ↑
#         先把 [1,2] 转成 [2,1]，才能和 x[2,2] 相乘
#
# 为什么这样存？
#   多个输出通道时更直观：[out, in] 一眼看出"有几个输出，每个输出用几个权重"
#   CNN 里的卷积核也是这个顺序：weight 形状 [out_ch, in_ch, kh, kw]


# ============================================================
# 三、坑：随机初始化的结果和你手写的对不上
# ============================================================
print("=" * 62)
print("二、随机初始化先看一眼")

y_random = layer(x)
print("随机权重下的输出 =", y_random.tolist())
# [[0.17137...], [0.91942...]]
print("对比正确答案 [[2],[4]] 的损失 =",
      round(((y_random - torch.tensor([[2.0], [4.0]])) ** 2).mean().item(), 4))
# 约 6.42，而你手写版是 0.68

# ★ 这不是库算错了！
#   是因为刚创建的层用随机数初始化权重，和你手写的 0.3/0.5 根本不是一组数
#   拿两个不同参数比结果，当然对不上
#   正确做法：把参数灌成一样的，再比


# ============================================================
# 四、把参数灌进去，让两边结果一致
# ============================================================
print("=" * 62)
print("三、灌入手写参数再比")

w_manual = torch.tensor([[0.3], [0.5]])     # 你的 [2,1]
b_manual = torch.tensor([0.1])
y_manual = x @ w_manual + b_manual

with torch.no_grad():
    # 为什么要 no_grad：
    #   直接改参数会被 autograd 拦截（和 A06 更新参数是同一个原因）
    #   copy_ 末尾的下划线表示"原地修改"
    layer.weight.copy_(w_manual.T)          # 转置成 [1,2] 再灌进去
    layer.bias.copy_(b_manual)

print("灌入后 weight =", layer.weight.tolist())     # [[0.3, 0.5]]
print("灌入后 bias   =", layer.bias.tolist())       # [0.1]

y_nn = layer(x)
print("手写 y =", y_manual.tolist())
print("nn   y =", y_nn.tolist())
print("完全相等?", torch.equal(y_manual, y_nn))
# 输出：
#   手写 y = [[1.399999976158142], [3.0]]
#   nn   y = [[1.399999976158142], [3.0]]
#   完全相等? True
#
# 那个 1.399999976158142 是 float32 精度问题（B03 讲过），不是错误


# ============================================================
# 五、验证"公式真的是 x @ weight.T + bias"
# ============================================================
print("=" * 62)
print("四、验证内部公式")

manual_formula = x @ layer.weight.T + layer.bias
print("按公式手算 =", manual_formula.tolist())
print("和 layer(x) 相等?", torch.equal(manual_formula, y_nn))

# ★ 所以 layer(x) 做的事就是一句话：x @ weight.T + bias
#   它把你 A08 里的两行（创建 w/b、算 x@w+b）打包成了一个对象


# ============================================================
# 六、这一层有几个参数
# ============================================================
print("=" * 62)
print("五、参数清点")

for name, p in layer.named_parameters():
    print(f"  {name:8} shape={tuple(p.shape)}  元素数={p.numel()}")

total = sum(p.numel() for p in layer.parameters())
print("总参数 =", total)      # 1*2 + 1 = 3

# 公式：(in_features + 1) * out_features = (2+1)*1 = 3
#   其中 "+1" 是每个输出配一个 bias


# ============================================================
# 七、layer(x) 为什么能像函数一样调用
# ============================================================
# layer 是个对象，但写 layer(x) 就能用，因为 nn.Module 实现了 __call__
#   __call__ 会先做内部准备（hook 等），再调用你自己写的 forward
#
# 所以：
#   layer(x)          <- 正规调用方式
#   layer.forward(x)  <- 能跑，但绕过了前面的准备，不要这么写


# ============================================================
# 总结
# ============================================================
# nn.Linear(in, out)       in = 输入特征数，out = 输出特征数
# weight 形状 [out, in]    ★ 是转置存的，和你手写的 [in, out] 相反
# bias   形状 [out]        每个输出一个
# 内部计算                  x @ weight.T + bias
# 参数量                    (in + 1) * out
# 随机初始化                每次创建权重不同，不能直接和你手写的比
# 灌参数                    with torch.no_grad(): layer.weight.copy_(w.T)
