import threading

latest_frame = None
lock = threading.Lock()

def update_frame(frame):
    """
    Called by stream_server.py to store the latest frame.
    """
    global latest_frame
    with lock:
        latest_frame = frame.copy()

    # DEBUG
    print("🔥 SHARED_CAMERA UPDATED")


def get_frame():
    """
    Called by ingest_worker.py to retrieve the latest frame.
    """
    global latest_frame
    with lock:
        frame_copy = None if latest_frame is None else latest_frame.copy()

    if frame_copy is None:
        print("❌ shared_camera: latest_frame is None")
    else:
        print("✅ shared_camera: frame output")

    return frame_copy
