# E05：保存、加载与恢复训练
#
# 【这一课解决什么问题】
#   训练了 10 个小时，机器崩了。想接着训，应该存什么？
#
#   多数人只存了模型权重。结果"续训"之后的模型和没中断时不一样，
#   而且你永远查不出为什么。
#
#   这一课讲清楚：权重、完整状态、推理用的模型，
#   是三件不同的事。


# ============================================================
# 一、两种需求，存的东西不一样
# ============================================================
#   需求 A：训练完了，拿去推理
#     需要：模型结构 + 权重 + 预处理参数 + 类别映射
#     不需要：优化器状态、epoch 数
#
#   需求 B：训练中断了，接着训
#     需要：上面全部 + 优化器状态 + epoch 数 + 学习率调度器的进度
#     不需要：多余的日志
#
# 把 A 当成 B 来用，就是"训练变慢了但说不出为什么"。


# ============================================================
# 二、state_dict 里到底有什么
# ============================================================
# 它不是整个模型对象，是一个字典：{参数名: 张量}
#
#   只含参数                {'0.weight', '0.bias', '2.weight', '2.bias'}
#   含 BatchNorm 的层       还会多出 'running_mean'、'running_var'、'num_batches_tracked'
#
# 那两个 running_xxx 是 buffer（B06 讲过）：不参与梯度，
# 但推理时要用（C07 讲过），所以必须一起存。
#
# ★ state_dict 里**不含**：
#   - 网络结构（那是 Python 类，存不进去）
#   - forward 函数
#   - 优化器状态
#   - 训练进度
#
# 所以加载时必须先自己把结构建出来，再把字典灌进去。


# ============================================================
# 三、实测：三种"恢复训练"的结果
# ============================================================
# 训练 20 步。对比三种做法：
#   A：一口气训完 20 步（基准）
#   B：训 10 步 -> 存"模型 + 优化器" -> 重建 -> 再训 10 步
#   C：训 10 步 -> 只存"模型"    -> 重建 -> 再训 10 步

import copy

import torch
import torch.nn as nn

torch.manual_seed(0)
X = torch.randn(64, 20)
y = torch.randint(0, 3, (64,))
loss_fn = nn.CrossEntropyLoss()


def make_model():
    torch.manual_seed(1)
    return nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Linear(32, 3))


def make_opt(model):
    return torch.optim.Adam(model.parameters(), lr=0.01)


def train(model, opt, steps):
    for _ in range(steps):
        opt.zero_grad()
        loss_fn(model(X), y).backward()
        opt.step()


# A：连续 20 步
m_a = make_model()
o_a = make_opt(m_a)
train(m_a, o_a, 20)
w_a = m_a[0].weight.detach().clone()

# B：10 步，存完整状态，续 10 步
m_b = make_model()
o_b = make_opt(m_b)
train(m_b, o_b, 10)
ckpt = {"model": copy.deepcopy(m_b.state_dict()),
        "optimizer": copy.deepcopy(o_b.state_dict())}
m_b2 = make_model()
o_b2 = make_opt(m_b2)
m_b2.load_state_dict(ckpt["model"])
o_b2.load_state_dict(ckpt["optimizer"])
train(m_b2, o_b2, 10)
w_b = m_b2[0].weight.detach().clone()

# C：10 步，只存模型，续 10 步
m_c = make_model()
o_c = make_opt(m_c)
train(m_c, o_c, 10)
w_ckpt = copy.deepcopy(m_c.state_dict())
m_c2 = make_model()
o_c2 = make_opt(m_c2)          # 优化器是全新的，Adam 的动量/方差全丢了
m_c2.load_state_dict(w_ckpt)
train(m_c2, o_c2, 10)
w_c = m_c2[0].weight.detach().clone()

print("=" * 62)
print("一、三种续训方式的结果")
print("A 连续训 20 步     权重前 3 =", [round(v, 6) for v in w_a[0][:3].tolist()])
print("B 存全量，续 10 步 权重前 3 =", [round(v, 6) for v in w_b[0][:3].tolist()])
print("C 只存权重，续 10 步 权重前 3 =", [round(v, 6) for v in w_c[0][:3].tolist()])
print()
print("B 和 A 完全一致?", torch.equal(w_a, w_b))
print("C 和 A 的最大差异 =", round((w_a - w_c).abs().max().item(), 6))

# 输出：
#   B 和 A 完全一致? True
#   C 和 A 的最大差异 = 0.107145
#
# ★ B 逐位相同，因为优化器状态也恢复了。
#   C 差了 0.1 —— 这已经是权重本身量级的差距了。
#
# 为什么？Adam 每步用的是"梯度的一阶矩和二阶矩"（B09 讲过）：
#   只恢复权重 -> 这两个历史量从头开始
#   -> 前几十步的更新方向和大小都和原来不同
#   -> 轨迹彻底分叉
#
# 用 SGD + momentum 也一样：momentum_buffer 丢了，
# 相当于"惯性"消失了，前几步会明显变慢。


# ============================================================
# 四、完整的 checkpoint 应该存什么
# ============================================================
# 一个能"真正续上"的检查点，至少要这几项：
#
#   model_state        模型参数 + buffer
#   optimizer_state    动量、Adam 的均值和方差、step 计数
#   epoch              训练到第几轮
#   scheduler_state    学习率调度器的进度（否则 lr 会从头开始）
#   best_metric        目前最好成绩（用来判断要不要覆盖 best）
#   config             超参数、数据路径、类别映射
#   rng_state          Python / NumPy / PyTorch 的随机数状态
#   version             代码版本、依赖版本
#
# 少存一项，续训结果就会偏一点。少存 optimizer_state 最明显。


# ============================================================
# 五、加载的正确顺序
# ============================================================
#   ① 先用**同样的结构定义**建一个模型
#        state_dict 里没有结构信息，结构必须由代码提供
#   ② load_state_dict(checkpoint['model_state'])
#   ③ 训练：还要恢复 optimizer、epoch、scheduler
#   ④ 推理：必须 model.eval()（C07 讲过，忘了就等于换了个模型）
#   ⑤ 预处理必须和训练时一模一样（C03 那份"约定"）
#
# 顺序错了的典型现象：
#   忘了 eval -> 准确率掉一截，还以为是权重坏了
#   预处理不一致 -> 推理结果全部错位，但程序不报错


# ============================================================
# 六、四个坑
# ============================================================
# ① 保存"最好的模型"时，别存引用
#    如果写 best = model.state_dict()，后面参数继续变，
#    best 里的张量可能是同一份，等你最后保存时已经不是当时的样子。
#    正确做法：立刻 deepcopy，或者当场 torch.save 到文件。
#
# ② 加载严格模式
#    load_state_dict(..., strict=True) 会检查键名和形状完全一致，
#    不一致直接报错。这是好事，别为了省事关掉。
#    如果确实要部分加载（比如只加载 backbone），
#    用 strict=False 之后**打印出哪些键没匹配上**，确认是你想要的。
#
# ③ 跨设备加载
#    在 GPU 上存的检查点，加载到只有 CPU 的机器要 map_location="cpu"，
#    否则会报错找不到 cuda。
#
# ④ 跨版本加载
#    torch.save 用的是 pickle，保存时连"类的路径"一起写进去了。
#    换了代码目录结构、改了类名，旧检查点可能加载不了。
#    这也是为什么"只存 state_dict"比"存整个模型对象"更可靠——
#    state_dict 只是字典和张量，不依赖你的类定义。


# ============================================================
# 七、这一课要记住的三句
# ============================================================
# 1. state_dict 只有参数和 buffer，没有结构、没有优化器、没有进度。
#
# 2. 只存权重去续训，结果一定和没中断过不一样
#    （实测差 0.1，权重本身就这个量级）。
#    要续训就必须存 optimizer_state 和 epoch。
#
# 3. 推理三件事必须和训练一致：结构、eval 模式、预处理。

print()
print("下一课：随机性与实验记录——为什么同样的代码跑两次结果不同。")
