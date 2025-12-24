# =========================================================
# embedder_pytorch_safe.py
# OSNet x0.25 MSMT17 ReID Embedder (FINAL)
# =========================================================

print("🧠 OSNet x0.25 embedder loaded", flush=True)

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchreid.reid.models import build_model


class TorchReIDEmbedder:
    def __init__(self, model_wts_path: str, device: str = "cuda"):
        print("🔧 Initializing OSNet x0.25 ReID Embedder", flush=True)

        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        print(f"🟢 Using device: {self.device}", flush=True)

        # --------------------------------------------------
        # Build OSNet backbone
        # --------------------------------------------------
        self.model = build_model(
            name="osnet_x0_25",
            num_classes=1041,   # MUST match MSMT17 checkpoint
            pretrained=False
        )

        if not os.path.isfile(model_wts_path):
            raise FileNotFoundError(f"❌ ReID weights not found: {model_wts_path}")

        print(f"📦 Loading OSNet weights: {model_wts_path}", flush=True)

        checkpoint = torch.load(model_wts_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        clean_state = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            # ❌ Drop classifier (not needed for ReID embeddings)
            if k.startswith("classifier"):
                continue
            clean_state[k] = v

        self.model.load_state_dict(clean_state, strict=False)

        # 🔥 Remove classifier entirely
        self.model.classifier = torch.nn.Identity()

        self.model.to(self.device)
        self.model.eval()

        # --------------------------------------------------
        # Preprocessing
        # --------------------------------------------------
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("✅ OSNet x0.25 embedder ready (classifier removed)", flush=True)

    def __call__(self, img_bgr: np.ndarray):
        if img_bgr is None or img_bgr.size == 0:
            return None

        img = cv2.resize(img_bgr, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor)

        emb = emb.squeeze().cpu().numpy()
        emb /= (np.linalg.norm(emb) + 1e-8)
        return emb
