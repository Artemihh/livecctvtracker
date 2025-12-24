# backend/app/utils/search_utils.py
import numpy as np
from typing import List, Dict, Any, Tuple

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Assumes a and b are 1D, L2-normalized or not — returns cosine in [-1,1]."""
    if a is None or b is None:
        return -1.0
    # ensure float32
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return -1.0
    return float(np.dot(a, b) / denom)

def search_top_k(query_emb: np.ndarray, candidates: List[Dict[str, Any]], top_k: int = 10, min_sim: float = 0.2) -> List[Tuple[Dict[str, Any], float]]:
    """
    Return list of (candidate_entry, similarity) sorted by similarity desc.
    candidate_entry must contain 'embedding' (numpy array).
    """
    if query_emb is None or len(candidates) == 0:
        return []

    sims = []
    for c in candidates:
        emb = c.get("embedding")
        if emb is None:
            continue
        sim = cosine_similarity(query_emb, emb)
        if sim >= min_sim:
            sims.append((c, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]
