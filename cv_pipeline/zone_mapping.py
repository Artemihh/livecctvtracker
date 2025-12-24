# cv_pipeline/zone_mapping.py
from pathlib import Path
import yaml

_config_cache = None


def load_config():
    """
    Load config.yaml once and cache it.
    """
    global _config_cache
    if _config_cache is None:
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as f:
            _config_cache = yaml.safe_load(f)
    return _config_cache


def get_zone_for_detection(camera_id: str, det) -> str | None:
    """
    Given camera_id and a detection dict with 'cx' and 'cy' (center in pixels),
    return the zone name from config.yaml or None if no zone matches.
    """
    cfg = load_config()
    zones_by_cam = cfg.get("zones", {})
    zones = zones_by_cam.get(camera_id, [])

    cx = det["cx"]
    cy = det["cy"]

    for z in zones:
        name = z["name"]
        x_min = z["x_min"]
        x_max = z["x_max"]
        y_min = z["y_min"]
        y_max = z["y_max"]

        if x_min <= cx <= x_max and y_min <= cy <= y_max:
            return name

    return None
