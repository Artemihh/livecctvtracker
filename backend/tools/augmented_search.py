# backend/tools/augmented_search.py
import cv2, requests, numpy as np, pprint, time
from pathlib import Path

URL = "http://127.0.0.1:8000/api/search/image"
IMG_PATH = Path(__file__).parent / "test_person.jpg"
PARAMS = {"top_k": 5, "min_sim": 0.15}   # low threshold for debugging

def augment_img(img):
    h, w = img.shape[:2]
    crops = []
    # original
    crops.append(img)
    # flipped
    crops.append(cv2.flip(img, 1))
    # center crop small/medium/large
    for scale in (0.9, 0.75, 0.6):
        nw, nh = int(w*scale), int(h*scale)
        x1 = max(0, (w - nw)//2)
        y1 = max(0, (h - nh)//2)
        crops.append(img[y1:y1+nh, x1:x1+nw])
    # small rotations
    for a in (-10, 10):
        M = cv2.getRotationMatrix2D((w/2,h/2), a, 1.0)
        crops.append(cv2.warpAffine(img, M, (w,h)))
    # resized variants
    for s in (0.8, 1.2):
        crops.append(cv2.resize(img, (0,0), fx=s, fy=s))
    return crops

def search_image_bytes(img_bgr):
    # encode to JPG bytes
    _, buf = cv2.imencode(".jpg", img_bgr)
    files = {"file": ("q.jpg", buf.tobytes(), "image/jpeg")}
    try:
        r = requests.post(URL, files=files, params=PARAMS, timeout=5)
        if r.status_code == 200:
            return r.json()
        else:
            return {"error_status": r.status_code, "text": r.text}
    except Exception as e:
        return {"error": str(e)}

def aggregate_results(results_list):
    # results_list: list of responses (dicts)
    agg = {}
    for res in results_list:
        if not isinstance(res, dict) or "matches" not in res:
            continue
        for m in res["matches"]:
            key = (m.get("camera_id"), m.get("track_id"))
            sim = float(m.get("similarity", 0.0))
            if key not in agg or sim > agg[key]["similarity"]:
                agg[key] = {**m, "similarity": sim}
    # sort
    sorted_matches = sorted(agg.values(), key=lambda x: x["similarity"], reverse=True)
    return sorted_matches

def main():
    img = cv2.imread(str(IMG_PATH))
    if img is None:
        print("test_person.jpg not found at", IMG_PATH)
        return
    variants = augment_img(img)
    print(f"Created {len(variants)} variants. Querying...")
    all_results = []
    for i, v in enumerate(variants):
        print("Query variant", i+1)
        res = search_image_bytes(v)
        pprint.pprint(res)
        all_results.append(res)
        time.sleep(0.2)
    agg = aggregate_results(all_results)
    print("\n=== Aggregated matches ===")
    pprint.pprint(agg[:10])

if __name__ == "__main__":
    main()
