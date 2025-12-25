from ultralytics import YOLO
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8n.pt").to(device)

def detect_persons(frame):
    results = model(frame, conf=0.5, classes=[0], verbose=False)
    boxes = []
    for r in results:
        for b in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, b.tolist())
            boxes.append((x1, y1, x2, y2))
    return boxes
