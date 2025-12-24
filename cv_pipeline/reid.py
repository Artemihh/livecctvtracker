# reid.py  -- simple prototype ReID using resnet50 global features
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
import numpy as np
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load a pretrained resnet50 and use its penultimate feature vector as embedding
_resnet = models.resnet50(pretrained=True)
_resnet.fc = nn.Identity()   # remove final fc
_resnet.eval()
_resnet.to(DEVICE)

# transform for person crop (H x W) typical ReID crop
_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 128)),   # Height x Width (common ReID ratio)
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225])
])

def get_embedding(bgr_crop):
    """
    Input: bgr_crop (numpy array, BGR) — a tight person crop (frame[y1:y2, x1:x2])
    Output: 1D numpy array (float32) normalized to unit length
    """
    if bgr_crop is None or bgr_crop.size == 0:
        return None

    # convert BGR->RGB for torchvision
    try:
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    except Exception:
        return None

    x = _transform(rgb).unsqueeze(0).to(DEVICE)   # (1,C,H,W)

    with torch.no_grad():
        feats = _resnet(x)                        # (1, 2048)
        feats = feats.cpu().numpy().reshape(-1)   # (2048,)

    # L2 normalize
    norm = np.linalg.norm(feats)
    if norm > 0:
        feats = feats / norm
    return feats.astype(np.float32)