# F05：导出失败与结果偏差的定位
#
# 【这一课解决什么问题】
#   导出这件事有两种坏结局，第二种更危险：
#     ① 直接报错，导不出来        -> 好办，至少你知道出事了
#     ② 导出成功，但语义被改了    -> 灾难，不报错，结果就是不对
#
#   这一课把三种典型陷阱和定位方法讲清楚。


# ============================================================
# 一、实测：三种写法，导出全都"成功"
# ============================================================
# 下面三个模型，导出时都没有抛异常，都生成了 onnx 文件。
# 但两个是坏的。

import os

import numpy as np
import torch
import onnxruntime as ort


class WithNumpy(torch.nn.Module):
    def forward(self, x):
        s = float(x.numpy().sum())     # 把张量转成 numpy 再算
        return x + s


class HardReshape(torch.nn.Module):
    def forward(self, x):
        return x.reshape(1, -1)        # 写死 batch=1


class DynamicCond(torch.nn.Module):
    def forward(self, x):
        if x.sum() > 0:                # 依赖张量值的 if
            return x * 2
        return x * 3


T = os.environ.get("TEMP", ".")
# 故意造一个 sum > 0 的样例输入，让导出时走 if 的第一个分支
x_trace = torch.ones(1, 3, 8, 8)

for name, model in [("用 numpy()", WithNumpy()),
                    ("写死 reshape", HardReshape()),
                    ("依赖值的 if", DynamicCond())]:
    path = os.path.join(T, "probe.onnx")
    try:
        torch.onnx.export(model, x_trace, path, dynamo=False, opset_version=17,
                          input_names=["x"], output_names=["y"])
        print(f"{name:<14} 导出成功")
    except Exception as e:
        print(f"{name:<14} 导出失败 -> {type(e).__name__}")

# 输出：三个全部"导出成功"。
#
# ★ 但"成功"不等于"对"。


# ============================================================
# 二、坏在哪：换一个输入就露馅
# ============================================================
print()
print("=" * 60)
print("二、换输入测试")

# ① 写死 reshape：喂 batch=2
m1 = HardReshape()
x2 = torch.randn(2, 3, 8, 8)
p1 = os.path.join(T, "a.onnx")
torch.onnx.export(m1, x_trace, p1, dynamo=False, opset_version=17,
                  input_names=["x"], output_names=["y"])
s1 = ort.InferenceSession(p1, providers=["CPUExecutionProvider"])
try:
    s1.run(None, {"x": x2.numpy()})
    print("写死 reshape 喂 batch=2: 成功")
except Exception as e:
    print("写死 reshape 喂 batch=2: 失败 ->", str(e)[:80])
# 失败：图里的 shape 被固化成了 [1, 24]，喂 2 张就对不上。
#   PyTorch 里这个模型明明能吃 batch=2（reshape(1,-1) 只是把总元素数变成 1 行）。

# ② 依赖值的 if：喂一个负数输入
m2 = DynamicCond()
p2 = os.path.join(T, "b.onnx")
torch.onnx.export(m2, x_trace, p2, dynamo=False, opset_version=17,
                  input_names=["x"], output_names=["y"])
s2 = ort.InferenceSession(p2, providers=["CPUExecutionProvider"])
test = torch.full((1, 3, 8, 8), -2.0)                # sum < 0，PyTorch 会走 *3
ref = m2(test).detach().numpy()
out = s2.run(None, {"x": test.numpy()})[0]
print("依赖值的 if  : PyTorch 形状", ref.shape, " ORT 形状", out.shape)
print("            最大差异 =", round(float(np.abs(ref - out).max()), 4))
print("            PyTorch 输出 =", float(ref.max()), " ORT 输出 =", float(out.max()))
# 输出：形状一样，但最大差异 2.0
#       PyTorch 输出 = -6.0（走了 *3）  ORT 输出 = -4.0（永远走 *2）
#
# 原因：导出时用 x_trace 跑了一遍，它的 sum > 0，所以图里**只记录 *2 这个分支**。
#       之后无论输入是什么，都只走 *2。
#       而 PyTorch 里遇到 sum < 0 会走 *3 —— 结果当然不一样。
#
# ★ 这就是最危险的故障：
#   不报错、形状正确、数值看起来也"像那么回事"，但逻辑全错。


# ============================================================
# 三、为什么会静默固化
# ============================================================
# 传统导出叫 "tracing"：拿一个样例输入真的跑一遍，
# **把你当下走过的路径记下来**。
#
#   走过的 if 分支 -> 只记这一支
#   reshape 的尺寸  -> 用当下的具体数字
#   .numpy() 的值   -> 当成常数写死在图里
#
# 所以导出时 PyTorch 会打印 TracerWarning：
#
#   "Converting a tensor to a Python boolean might cause the trace
#    to be incorrect... treated as a constant in the future."
#
# ★ 看到 TracerWarning 就要停下来查。它不是"警告可以忽略"，
#   它是在说"这个模型的语义没法被完整记录下来"。


# ============================================================
# 四、定位方法一：最小复现
# ============================================================
# 导出失败时，不要对着整个 YOLO 调。做法是"二分"：
#
#   ① 只导出第一个模块（比如一个 Conv+BN+SiLU），能过吗？
#   ② 加上第二个模块，能过吗？
#   ③ 一直加到你怀疑的那一层，它开始失败 -> 问题就在这层
#
# 这种方法能把"300 层的模型报错"缩小到"某一个算子不支持"。


# ============================================================
# 五、定位方法二：三段隔离
# ============================================================
# 导出成功但结果不对，问题可能在三段中的任意一段：
#
#   预处理  ->  模型  ->  后处理
#
# 办法是**先把预处理和后处理排除掉**：
#
#   ① 保存 PyTorch 里那份已经预处理好的张量（numpy 存盘）
#   ② 两边都用这同一份张量当输入
#      PyTorch: 加载张量 -> 模型 -> 输出 A
#      ORT    : 加载张量 -> 模型 -> 输出 B
#   ③ 比较 A 和 B
#
#   如果 A ≈ B   -> 模型没问题，问题在预处理或后处理
#   如果 A ≠ B   -> 模型导坏了，回去查算子
#
# ★ 这一步必须做。不做的话，你会花几个小时怀疑模型，
#   结果最后发现是"推理时忘了除以 255"。


# ============================================================
# 六、结果偏差的四个来源（按概率排序）
# ============================================================
#   ① 预处理不一致        最常见。少除 255、BGR/RGB、忘了标准化
#   ② 后处理不一致        坐标逆变换少减 padding、阈值不同
#   ③ 语义被固化          上面讲的 if / reshape / numpy
#   ④ 精度差异            FP16 导出、量化；这类差异有明确量级（F02）
#
# 排查顺序就是按这个顺序，从最常见开始。


# ============================================================
# 七、检查清单
# ============================================================
# 导出前：
#   □ forward 里有没有依赖输入数值的 if / for
#   □ 有没有 .numpy() / .item() / float() 把张量转成 Python 数
#   □ 有没有把 batch 或尺寸写死的 reshape / view
#   □ 有没有打印语句（有些会被固化，有些会拖慢导出）
#   □ 模型是不是 eval() 模式
#
# 导出后：
#   □ 用同一份输入张量对拍
#   □ 换 2~3 个不同输入再对拍（批次、尺寸、边界值）
#   □ 相对误差在精度量级内、结论（argmax / 框数量）一致


# ============================================================
# 八、这一课要记住的三句
# ============================================================
# 1. "导出成功"不等于"导出正确"。依赖数值的 if 会被固化成一个分支，
#    不报错，但换个输入结果就错（实测：PyTorch -6，ORT -4）。
#
# 2. TracerWarning 不是提醒，是"这个模型的语义没法完整记录"的警告。
#
# 3. 结果不对时，先把预处理和后处理排除：
#    用同一份张量喂两边，只比模型的输出。

print()
print("下一课：把前面串起来，做一个能交付的推理程序。")
