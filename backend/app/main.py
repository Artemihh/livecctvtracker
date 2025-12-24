from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torchreid.reid.models import build_model

# ---------------- PATHS ----------------
BASE_DIR = Path(__file__).resolve().parents[2]
CV_DIR = BASE_DIR / "cv_pipeline"
UPLOAD_DIR = CV_DIR / "uploaded"
UPLOAD_DIR.mkdir(exist_ok=True)

PROBE_EMB_PATH = CV_DIR / "probe_embedding.npy"
WEIGHTS_PATH = CV_DIR / "osnet_x0_25_msmt17.pt"

# ---------------- STATE ----------------
PROBE_RESULT = {
    "matched": False,
    "person_id": None,
    "similarity": None,
}

# ---------------- FASTAPI ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- MODEL ----------------
class ProbeEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = build_model(
            name="osnet_x0_25",
            num_classes=1000,
            pretrained=False,
        )

        ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        clean = {k.replace("module.", ""): v for k, v in state.items()
                 if not k.startswith("classifier")}

        self.model.load_state_dict(clean, strict=False)
        self.model.classifier = torch.nn.Identity()
        self.model.to(self.device).eval()

        self.tf = T.Compose([
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def extract(self, img_path):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(x).squeeze().cpu().numpy()
        return emb / (np.linalg.norm(emb) + 1e-8)

embedder = ProbeEmbedder()

# ---------------- ROUTES ----------------
@app.post("/upload_probe")
async def upload_probe(file: UploadFile = File(...)):
    global PROBE_RESULT
    PROBE_RESULT = {"matched": False, "person_id": None, "similarity": None}

    img_path = UPLOAD_DIR / file.filename
    with open(img_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    emb = embedder.extract(img_path)
    np.save(PROBE_EMB_PATH, emb)

    print("📸 Probe uploaded & embedding saved", flush=True)
    return {"status": "probe_loaded"}

@app.get("/probe_status")
def probe_status():
    return PROBE_RESULT

@app.post("/probe_match")
def probe_match(payload: dict):
    global PROBE_RESULT
    PROBE_RESULT = payload
    print("🎯 MATCH RECEIVED FROM WORKER:", payload, flush=True)
    return {"ok": True}
