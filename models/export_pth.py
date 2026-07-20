"""
第一步：把完整模型对象转存为 state_dict .pth
必须和 deeplob.py / axiallob.py / vitlob.py 放在同一目录下运行
"""
import torch
from deeplob  import DeepLOB
from axiallob import AxialLOB
from vitlob   import ViT_1D

# DeepLOB
model = torch.load("deeplob_model", map_location="cpu", weights_only=False)
model.eval()
torch.save(model.state_dict(), "deeplob_best.pth")
print("Saved deeplob_best.pth")

# AxialLOB
model = torch.load("axiallob_model", map_location="cpu", weights_only=False)
model.eval()
torch.save(model.state_dict(), "axiallob_best.pth")
print("Saved axiallob_best.pth")

# ViTLOB
model = torch.load("Vit-LOB_model.pt", map_location="cpu", weights_only=False)
model.eval()
torch.save(model.state_dict(), "vitlob_best.pth")
print("Saved vitlob_best.pth")