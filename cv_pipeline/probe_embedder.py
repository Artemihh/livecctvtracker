import cv2
import torch
import numpy as np
from torchreid.reid.models import build_model
import torchvision.transforms as T


class ProbeEmbedder:
    def __init__(self, weights_path, device="cuda"):
        self.device = torch.device(
            "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        )

        self.model = build_model(
            name="osnet_x0_25",
            num_classes=1000,
            pretrained=False
        )

        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

        clean = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            if not k.startswith("classifier"):
                clean[k] = v

        self.model.load_state_dict(clean, strict=False)
        self.model.classifier = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError("Cannot read probe image")

        img = cv2.resize(img, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(tensor)

        emb = emb.squeeze().cpu().numpy()
        emb /= (np.linalg.norm(emb) + 1e-8)
        return emb
