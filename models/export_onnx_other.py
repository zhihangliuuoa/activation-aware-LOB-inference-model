"""
Export DeepLOB / AxialLOB / ViTLOB  →  .pth (state_dict) + .onnx
把本脚本和以下文件放在同一目录下运行：
    axiallob_model
    deeplob_model
    Vit-LOB_model.pt
    axiallob.py  /  deeplob.py  /  vitlob.py   (模型定义)

依赖：
    pip install torch onnx einops
"""

import deeplob  as deeplob_module   # noqa: F401
import axiallob as axiallob_module  # noqa: F401
import vitlob   as vitlob_module 

import torch
import onnx
from deeplob  import DeepLOB
from axiallob import AxialLOB
from vitlob   import ViT_1D


def export(name: str, model: torch.nn.Module,
           dummy: torch.Tensor, pth_out: str, onnx_out: str):

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # 1. Export ONNX
    torch.onnx.export(
        model,
        dummy,
        onnx_out,
        opset_version=17,        # JetPack 6 + TRT 8.6+
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,       # fixed batch=1
    )
    print(f"  Saved onnx : {onnx_out}")

    # 2. Verify
    m = onnx.load(onnx_out)
    onnx.checker.check_model(m)
    inp_dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    out_dims = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    print(f"  Input  shape : {inp_dims}")
    print(f"  Output shape : {out_dims}")
    print(f"  ✓ ONNX check passed")


# ─────────────────────────────────────────────────────────
# DeepLOB
# ─────────────────────────────────────────────────────────
# Step 1: load full model object, save state_dict
model_obj = torch.load("deeplob_model", map_location="cpu", weights_only=False)
model_obj.eval()
torch.save(model_obj.state_dict(), "deeplob.pth")
print("Saved deeplob.pth")

# Step 2: rebuild model, load state_dict, export
model = DeepLOB(num_classes=3, device="cpu")
model.load_state_dict(torch.load("deeplob.pth", map_location="cpu"))
model.eval()
export("DeepLOB", model, torch.randn(1, 1, 100, 40), "deeplob.pth", "deeplob.onnx")


# ─────────────────────────────────────────────────────────
# AxialLOB
# ─────────────────────────────────────────────────────────
model_obj = torch.load("axiallob_model", map_location="cpu", weights_only=False)
model_obj.eval()
torch.save(model_obj.state_dict(), "axiallob.pth")
print("Saved axiallob.pth")

model = AxialLOB(W=40, H=40, c_in=32, c_out=32, c_final=4,
                 n_heads=4, pool_kernel=(1, 4), pool_stride=(1, 4), num_classes=3)
model.load_state_dict(torch.load("axiallob.pth", map_location="cpu"))
model.eval()
export("AxialLOB", model, torch.randn(1, 1, 40, 40), "axiallob.pth", "axiallob.onnx")


# ─────────────────────────────────────────────────────────
# ViTLOB
# ─────────────────────────────────────────────────────────
model_obj = torch.load("Vit-LOB_model.pt", map_location="cpu", weights_only=False)
model_obj.eval()
torch.save(model_obj.state_dict(), "vitlob.pth")
print("Saved vitlob.pth")

model = ViT_1D(seq_len=50, patch_size=10, num_classes=3,
               dim=64, depth=6, heads=8, mlp_dim=3,
               channels=40, dim_head=8, dropout=0.1, emb_dropout=0.1)
model.load_state_dict(torch.load("vitlob.pth", map_location="cpu"))
model.eval()
export("ViTLOB", model, torch.randn(1, 40, 50), "vitlob.pth", "vitlob.onnx")


print("\n" + "="*60)
print("All done. Files generated:")
print("  deeplob.pth  /  deeplob.onnx")
print("  axiallob.pth /  axiallob.onnx")
print("  vitlob.pth   /  vitlob.onnx")
print()
print("Copy *.onnx to Jetson Orin Nano, then:")
print("  trtexec --onnx=deeplob.onnx  --saveEngine=deeplob.engine  --fp16")
print("  trtexec --onnx=axiallob.onnx --saveEngine=axiallob.engine --fp16")
print("  trtexec --onnx=vitlob.onnx   --saveEngine=vitlob.engine   --fp16")