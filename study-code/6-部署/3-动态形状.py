# F03：动态 batch 与动态尺寸
#
# 【这一课解决什么问题】
#   导出时形状是写死的：[1, 3, 640, 640]。
#   但部署时用户可能一次给 1 张、也可能给 3 张；图片可能 640 高，也可能 480。
#
#   这一课讲清楚两件事：
#     ① 声明"动态"到底在声明什么
#     ② 为什么很多部署干脆固定尺寸（YOLO 的 LetterBox 就是这么来的）


# ============================================================
# 一、"动态"要分别声明
# ============================================================
# 一个 [1, 3, 640, 640] 的输入有四个轴：
#
#   axis 0    batch       一次几张图
#   axis 1    channel     通常是 3，基本不变
#   axis 2    height      图片高度
#   axis 3    width       图片宽度
#
# 导出时可以把任意一个轴声明成动态（不给固定数字）。
#
# ★ 关键：**动态 batch 和动态尺寸是两件独立的事。**
#   只声明 batch 动态，输入高度宽度仍然必须是 640。
#   这点非常容易混。


# ============================================================
# 二、实测：声明了动态之后
# ============================================================

import os

import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

torch.manual_seed(0)

# 版本 A：用自适应池化收尾，对尺寸不敏感
model_a = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),        # 不管输入多大，输出都是 1x1
    nn.Flatten(), nn.Linear(64, 10),
).eval()

x = torch.randn(1, 3, 128, 128)
tmp = os.environ.get("TEMP", ".")
path_a = os.path.join(tmp, "dyn_a.onnx")
path_s = os.path.join(tmp, "static.onnx")

with torch.no_grad():
    torch.onnx.export(model_a, x, path_s, dynamo=False, opset_version=17,
                      input_names=["images"], output_names=["logits"])
    torch.onnx.export(model_a, x, path_a, dynamo=False, opset_version=17,
                      input_names=["images"], output_names=["logits"],
                      dynamic_axes={"images": {0: "batch", 2: "h", 3: "w"},
                                    "logits": {0: "batch"}})

sess_s = ort.InferenceSession(path_s, providers=["CPUExecutionProvider"])
sess_a = ort.InferenceSession(path_a, providers=["CPUExecutionProvider"])

print("=" * 60)
print("一、两种模型接受什么样输入")
print("静态模型的输入声明:", sess_s.get_inputs()[0].shape)   # [1, 3, 128, 128]
print("动态模型的输入声明:", sess_a.get_inputs()[0].shape)   # ['batch', 3, 'h', 'w']

for shape in [(1, 3, 96, 96), (1, 3, 160, 160), (4, 3, 128, 128)]:
    arr = np.random.randn(*shape).astype(np.float32)
    try:
        out = sess_a.run(None, {"images": arr})[0]
        print(f"  动态模型喂 {shape} -> 输出 {out.shape}  OK")
    except Exception as e:
        print(f"  动态模型喂 {shape} -> 失败 {str(e)[:60]}")

# 输出：
#   动态模型喂 (1, 3, 96, 96)   -> 输出 (1, 10)  OK
#   动态模型喂 (1, 3, 160, 160) -> 输出 (1, 10)  OK
#   动态模型喂 (4, 3, 128, 128) -> 输出 (4, 10)  OK
#
# 三种形状都吃得下。为什么？
# 因为 AdaptiveAvgPool2d(1) 不管特征图多大，都压成一个数。
# 后面的 Linear 永远看到 64 个特征。
#
# ★ 这是"结构上支持动态尺寸"的关键：
#   网络的最后一段不能对尺寸有硬性假设。


# ============================================================
# 三、结构不支持，声明了也没用
# ============================================================
# 换成最常见的写法：Flatten + 固定大小的 Linear。
# 照样声明动态尺寸，然后喂一个不同大小的图：

model_b = nn.Sequential(
    nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(8 * 64 * 64, 10),      # 写死：128/2 = 64
).eval()

path_b = os.path.join(tmp, "dyn_b.onnx")
with torch.no_grad():
    torch.onnx.export(model_b, x, path_b, dynamo=False, opset_version=17,
                      input_names=["images"], output_names=["logits"],
                      dynamic_axes={"images": {0: "batch", 2: "h", 3: "w"}})

sess_b = ort.InferenceSession(path_b, providers=["CPUExecutionProvider"])
try:
    sess_b.run(None, {"images": np.random.randn(1, 3, 96, 96).astype(np.float32)})
    print("  固定 Linear 版喂 96x96: OK")
except Exception as e:
    print("  固定 Linear 版喂 96x96: 失败 ->", str(e)[:110])

# 输出：Gemm 维度不匹配，W:{10, 32768} K: 18432
#
#   32768 = 8 × 64 × 64   （128 输入时的特征数）
#   18432 = 8 × 48 × 48   （96 输入时的特征数）
#
# ★ 结论：**动态尺寸不是"导出时声明一下"就行的，
#   网络结构本身必须没有"写死大小"的环节。**
#   声明动态只是允许引擎接受不同形状，
#   能不能算通，取决于图里有没有按固定尺寸写死的运算。


# ============================================================
# 四、动态的代价
# ============================================================
# 实测（这个模型，CPU，batch=1）：
#   静态模型 4.24 ms
#   动态模型 4.60 ms     约慢 8%
#
# 注意：这个数抖动很大（不同次测量能差一倍），
# 在小模型上基本是噪声。所以准确的说法是：
#
#   ★ 在 ONNX Runtime 的 CPU 上，动态带来的开销通常很小。
#
#   真正的代价出现在"预编译型引擎"上，比如 TensorRT：
#   它会针对固定形状预先挑算法、分配显存、融合算子，
#   形状一变就要重新构建（可能要几十秒）。
#
#   所以：
#     通用推理（ORT）        动态一般可以接受
#     极致性能（TensorRT）    强烈建议固定形状
#     嵌入式/移动端            固定形状，省内存省电


# ============================================================
# 五、YOLO 为什么固定 640：LetterBox
# ============================================================
# 检测模型比分类模型更"怕"尺寸变化，原因有三层：
#
#   ① 结构上：网络做了 32 倍下采样，输入必须是 32 的倍数，
#      否则最后特征图的尺寸会有小数，除不尽。
#
#   ② 输出上：检测头输出的位置数是 80×80 + 40×40 + 20×20 = 8400。
#      输入尺寸一变，这个数就变，后处理代码也得跟着改。
#
#   ③ 后处理上：框的坐标是在"输入图坐标系"里的，
#      要和原图对应，就必须记住缩放比例和填充量。
#
# 所以工业界的做法是：**把任意尺寸的图，统统变形到 640×640**，
# 这就是 LetterBox：
#
#   等比缩放，让长边变成 640
#   短边不够的部分用灰色（通常是 114）补边
#   -> 长宽比保持不变，物体不会被拉变形
#
# 推理完再逆变换：
#   把框的坐标减掉 padding、除以缩放比例
#   -> 回到原图的坐标系
#
# ★ 关键点：LetterBox 的 ratio 和 padding 必须和后处理共享，
#   否则框会整体偏移（G02 会细讲这套坐标变换）。
#
#   为什么必须保持长宽比？
#   如果直接把 1280×720 硬拉成 640×640，人会被拉成瘦子，
#   模型没见过这种形状，准确率会掉。


# ============================================================
# 六、这一课要记住的三句
# ============================================================
# 1. 动态 batch 和动态尺寸是两回事，要分别声明。
#    只声明 batch，高度宽度仍然是固定的。
#
# 2. 声明动态不等于结构支持。图里只要有"按固定尺寸写死的运算"
#    （典型：Flatten + 固定 Linear），换个尺寸就会报错。
#    想真支持，最后一段要用自适应池化这类尺寸无关的操作。
#
# 3. 检测模型通常固定输入尺寸 + LetterBox。
#    因为下采样要求 32 的倍数，输出位置数随尺寸变，后处理也要跟着变。

print()
print("下一课：性能怎么公平地测——延迟、吞吐、预热、异步。")
