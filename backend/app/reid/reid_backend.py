# backend/app/reid/reid_backend.py

import torch
import torchvision.transforms as T
from torchvision.models import resnet50
import numpy as np
import cv2

# -------------------------------------------------
# DEVICE (GPU / CPU)
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔵 ReID DEVICE:", DEVICE)

# -------------------------------------------------
# MODEL (LOAD ONCE)
# -------------------------------------------------
_model = None

def _load_model():
    model = resnet50(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Identity()   # 2048‑D embedding
    model.eval()
    model.to(DEVICE)                # 🔥 GPU MOVE
    return model

def get_model():
    global _model
    if _model is None:
        _model = _load_model()
    return _model

# -------------------------------------------------
# TRANSFORM
# -------------------------------------------------
_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# -------------------------------------------------
# EMBEDDING FUNCTION
# -------------------------------------------------
@torch.no_grad()
def get_embedding_from_bgr(frame_bgr: np.ndarray):
    model = get_model()

    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transform(img).unsqueeze(0).to(DEVICE)   # 🔥 GPU INPUT

    feat = model(tensor)
    feat = torch.nn.functional.normalize(feat, dim=1)

    return feat.squeeze(0).cpu().numpy()   # back to CPU numpy
