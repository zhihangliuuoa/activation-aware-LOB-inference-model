import torch
from CFLOB import ConvFormerLOB

model = ConvFormerLOB(
    window_len=10,
    patch_size=(1, 40),
    embed_dim=4,
    num_heads=4,
    mlp_ratio=4,
    num_classes=3
)

# 第一步是把 torch.save 的模型改成 pth. 以后训练时也可直接保存为 pth
model = torch.load("cflob_best", weights_only=False)
model.eval()
torch.save(model.state_dict(), "cflob_best.pth")

# 第二步，导出结果文件为 cflob.onnx 和 cflob.onnx.data
state = torch.load("cflob_best.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()

dummy = torch.randn(1, 1, 10, 40)

torch.onnx.export(
    model,
    dummy,
    "cflob.onnx",
    opset_version=17,          # JP6 + TRT8.6+ 推荐
    do_constant_folding=True,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes=None          # ⚠️ 固定 batch=1
)
