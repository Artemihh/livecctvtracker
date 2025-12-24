import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchreid.reid.models import build_model


class ProbeEmbedder:
    def __init__(self, weights_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = build_model(
            name="osnet_x0_25",
            num_classes=1000,
            pretrained=False
        )

        checkpoint = torch.load(weights_path, map_location="cpu")
        state = checkpoint.get("state_dict", checkpoint)

        clean = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[7:]
            if not k.startswith("classifier"):
                clean[k] = v

        self.model.load_state_dict(clean, strict=False)
        self.model.classifier = torch.nn.Identity()
        self.model.to(self.device).eval()

        self.tf = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def extract(self, path):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError("Probe image not readable")

        img = cv2.resize(img, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        x = self.tf(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model(x).squeeze().cpu().numpy()

        emb /= (np.linalg.norm(emb) + 1e-8)
        return emb
