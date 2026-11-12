import torch

# 这次来学习最小的训练循环

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
target = torch.tensor([[2.0], [4.0]], dtype=torch.float64)

w = torch.tensor([[0.3], [0.5]], dtype=torch.float64, requires_grad=True)
b = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)

lr = 0.01
history = []
# 还是先照旧创建

for step in range(201):
    # 每轮清梯度
    w.grad = None
    b.grad = None

    # 计算预测和损失
    y = x @ w + b
    loss = ((y - target) ** 2).mean()

    # 算梯度
    loss.backward()

    # 更新参数
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    history.append(loss.item()) # 存储历史的损失率

    if step % 20 == 0: # 每20轮输出显示损失和参数
        print(f"step {step:3d}  loss = {loss.item():.6f}  "
              f"w = [{w[0,0]:.4f}, {w[1,0]:.4f}]  b = [{b[0]:.4f}]")

print("=" * 55)
print("训练结束")
print("最终 w =", [round(v, 4) for v in w.flatten().tolist()],
      " b =", [round(v, 4) for v in b.flatten().tolist()])
print("最终 loss =", round(history[-1], 6))

# ============================================================
# 四步顺序不能乱，漏一步会怎样
# ============================================================
# 漏①清梯度  ->  梯度越累越大，等效学习率越来越猛，参数飞掉
# 漏②前向    ->  没有 loss，backward 无从开始
# 漏③反向    ->  w.grad 是 None，更新时直接报错
# 漏④更新    ->  参数永远不变，loss 一动不动

# ============================================================
# 三个坑
# ============================================================
# 1. 日志存 loss.item()，不要存 loss 张量（带计算图，内存不释放）
# 2. loss 不一定每一步都下降。本例恰好单调下降，但真实训练会上下波动，
#    看的是几十步的整体趋势，不是相邻两步
# 3. 别指望训到 loss = 0。那通常意味着参数被推极端、数据太少或数据泄漏