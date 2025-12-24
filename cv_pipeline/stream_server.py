import cv2
from flask import Flask, Response

app = Flask(__name__)

# ==============================
# Camera setup
# ==============================
CAMERA_INDEX = 0  # change to video file path if needed
camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    raise RuntimeError("❌ Cannot access camera")

# ==============================
# MJPEG frame generator
# ==============================
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame_bytes
            + b"\r\n"
        )

# ==============================
# Routes
# ==============================
@app.route("/feed")
def feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/")
def health():
    return {"status": "MJPEG stream running"}

# ==============================
# Server entry point
# ==============================
if __name__ == "__main__":
    print("🚀 MJPEG stream running at http://127.0.0.1:5001/feed")
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False,
        threaded=True
    )

