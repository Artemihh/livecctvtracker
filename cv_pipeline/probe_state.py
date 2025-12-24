# cv_pipeline/probe_state.py
import numpy as np
import os

PROBE_FILE = "cv_pipeline/probe_embedding.npy"

def save_probe(embedding):
    np.save(PROBE_FILE, embedding)

def load_probe():
    if not os.path.exists(PROBE_FILE):
        return None
    return np.load(PROBE_FILE)
