# tracking.py (DeepSORT wrapper — tuned for stability)
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortWrapper:
    def __init__(self,
                 max_age=30,
                 n_init=3,
                 max_cosine_distance=0.25,
                 nms_max_overlap=0.6,
                 embedder="mobilenet"):
        """
        Tuned values:
         - n_init=3 (require 3 confirmations before a track is confirmed)
         - max_cosine_distance lower (0.25) to reduce wrong matches
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            max_cosine_distance=max_cosine_distance,
            embedder=embedder
        )

    def update(self, frame_bgr, detections):
        """
        detections: list of dicts with keys x1,y1,x2,y2,conf,cls
        Returns list of tracked dicts:
        {
          'track_id','cls','conf','x1','y1','x2','y2','w','h','cx','cy'
        }
        """
        ds_inputs = []
        for d in detections:
            bbox = [int(d["x1"]), int(d["y1"]), int(d["x2"]), int(d["y2"])]
            ds_inputs.append((bbox, float(d.get("conf", 0.0)), d.get("cls", "obj")))

        tracks = self.tracker.update_tracks(ds_inputs, frame=frame_bgr)

        tracked = []
        for t in tracks:
            # t.is_confirmed() exists in deep-sort-realtime
            if not t.is_confirmed():
                continue
            tid = t.track_id
            l, t0, r, b = t.to_ltrb()
            x1, y1, x2, y2 = int(l), int(t0), int(r), int(b)
            w = x2 - x1
            h = y2 - y1
            cx = x1 + w / 2
            cy = y1 + h / 2

            # the library stores detection class and conf if provided
            cls = t.get_det_class() or "unknown"
            conf = float(t.get_det_conf() or 0.0)

            tracked.append({
                "track_id": int(tid),
                "cls": cls,
                "conf": conf,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "w": w, "h": h,
                "cx": cx, "cy": cy
            })
        return tracked
