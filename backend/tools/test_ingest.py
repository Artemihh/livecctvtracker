# quick snippet to POST and print full server response
import requests, time, json

url = "http://127.0.0.1:8000/api/ingest"
payload = {
  "detections": [
    {
      "camera_id": "CAM_TEST",
      "track_id": 1,
      "cls": "person",
      "bbox": [10, 20, 100, 200],
      "cx": 55.0, "cy": 110.0, "w": 90.0, "h": 180.0,
      "timestamp": time.time(),
      "reid_embedding": [0.1, 0.2, 0.3]  # small example
    }
  ]
}
r = requests.post(url, json=payload, timeout=5)
print("status:", r.status_code)
try:
    print("body:", json.dumps(r.json(), indent=2))
except Exception:
    print("text:", r.text)
