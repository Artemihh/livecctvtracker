import os
import cv2
import torch
import gradio as gr
import numpy as np

from cv_pipeline.yolo_detector import detect_persons
from torchreid.utils import FeatureExtractor

# =========================
# CONFIG
# =========================
SIM_THRESHOLD = 0.60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROBE_PATH = "probe_embedding.npy"

# =========================
# CLEAN START (NO CACHE)
# =========================
if os.path.exists(PROBE_PATH):
    os.remove(PROBE_PATH)

# =========================
# LOAD REID MODEL
# =========================
extractor = FeatureExtractor(
    model_name="osnet_x0_25",
    model_path="cv_pipeline/osnet_x0_25_msmt17.pt",
    device=DEVICE
)
print("🟢 ReID model loaded on", DEVICE)

# =========================
# CAMERA (WINDOWS SAFE)
# =========================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("❌ Cannot open camera")

# =========================
# HELPERS
# =========================
def l2(x):
    return x / (np.linalg.norm(x) + 1e-8)

# =========================
# PROBE UPLOAD
# =========================
def upload_probe(img):
    if img is None:
        return "❌ No image uploaded"

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    boxes = detect_persons(img)
    if len(boxes) == 0:
        return "❌ No person detected in probe image"

    x1, y1, x2, y2 = boxes[0]
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return "❌ Invalid crop"

    emb = extractor(crop)[0]
    emb = emb.detach().cpu().numpy()
    emb = l2(emb)

    np.save(PROBE_PATH, emb)
    return "✅ Probe registered. Live matching started."

# =========================
# LIVE STREAM (SAFE)
# =========================
def live_feed():
    while True:
        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            status = "⚠️ Upload probe image first"
            probe = None

            if os.path.exists(PROBE_PATH):
                probe = l2(np.load(PROBE_PATH))
                status = "❌ NO MATCH"

            boxes = detect_persons(frame)

            for (x1, y1, x2, y2) in boxes:
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                emb = extractor(crop)[0]
                emb = emb.detach().cpu().numpy()
                emb = l2(emb)

                if probe is not None:
                    sim = float(np.dot(emb, probe))

                    if sim >= SIM_THRESHOLD:
                        status = f"✅ MATCH FOUND (Similarity: {sim:.3f})"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(
                            frame,
                            f"MATCH {sim:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,255,0),
                            2
                        )
                        break
                    else:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255,255,0), 2)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield frame, status

        except Exception as e:
            print("⚠️ Live feed error:", e)
            continue

# =========================
# GRADIO UI
# =========================
with gr.Blocks(title="Live CCTV Person Re-Identification") as app:
    gr.Markdown("## 🎥 Live CCTV Person Re-Identification")

    with gr.Row():
        with gr.Column(scale=1):
            probe_img = gr.Image(label="Upload Probe Image")
            probe_btn = gr.Button("Register Probe")
            probe_status = gr.Textbox(label="Status")

        with gr.Column(scale=2):
            video = gr.Image(label="Live Feed", streaming=True)
            match_status = gr.Textbox(label="Match Result")

    probe_btn.click(upload_probe, probe_img, probe_status)
    app.load(live_feed, None, [video, match_status])

app.launch()
