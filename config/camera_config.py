"""
Per-camera configuration.

Each camera gets its own config block. This is intentionally a plain Python
dict/JSON structure right now — later this can be loaded from a database or
exposed through a settings UI without changing any downstream code.
"""

import json
from pathlib import Path

# Motion-area thresholds (in pixels) — lower = trips on smaller movements
SENSITIVITY_PRESETS = {
    "high_security": {"motion_area_threshold": 300},
    "normal": {"motion_area_threshold": 1500},
    "low": {"motion_area_threshold": 4000},
}

DEFAULT_CONFIG = {
    "camera_id": "default_cam",
    "source": 0,  # 0 = default webcam; else RTSP URL or DroidCam URL
    "schedule": {
        # list of [start, end] 24h "HH:MM" windows where full AI pipeline runs
        # outside these windows -> plain CCTV recording mode (motion save only)
        "ai_mode": [["00:00", "23:59"]]
    },
    "sensitivity": "normal",
    "motion_area_threshold": 1500,  # the actual live number (pixels) - the
    # sensitivity preset above is just a quick way to set this; it can also
    # be tuned directly. Lower = trips on smaller movements.
    "hazard_confidence_threshold": 0.2,   # min YOLO confidence to call something a fire/smoke hazard
    "face_match_threshold": 0.45,          # min cosine similarity to call a face a match
    "person_confidence_threshold": 0.4,    # min person-detection confidence before the (expensive) face/identity pipeline runs at all
    "ignore_pets": False,
    "ignore_zones": [],  # list of polygons, each a list of [x, y] points - motion INSIDE these is ignored
    "detection_zones": [[
        [0, 520],
        [200, 520],
        [200, 720],
        [0, 720]
    ]],  # list of polygons - if set, person/identity pipeline ONLY counts motion whose center falls inside one of these (e.g. a doorway). Empty = whole frame counts.
    "auth_schedule": {
        "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "start": "00:00",
        "end": "23:59",
    },
    "cooldown_seconds": 2.5,      # stop recording after this much no-motion
    "max_clip_seconds": 45,       # hard cap on a single clip's length
    "pre_buffer_seconds": 1.5,    # rolling buffer captured before trigger
    "repetitive_motion_window": 60,   # seconds
    "repetitive_motion_max_triggers": 5,
    "hazard_check_interval": 2.0,  # seconds between fire/smoke checks - the
    # capture loop self-throttles (see core/capture.py) so a check that
    # takes longer than this to run is never overlapped by the next one.
}


def load_camera_config(path: str) -> dict:
    """Load a camera config JSON file, filling in any missing keys with defaults."""
    cfg = dict(DEFAULT_CONFIG)
    p = Path(path)
    if p.exists():
        with open(p, "r") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)
    return cfg


def save_default_config(path: str):
    """Write out a starter config file the user can edit."""
    with open(path, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)


def get_motion_threshold(cfg: dict) -> int:
    if "motion_area_threshold" in cfg:
        return cfg["motion_area_threshold"]
    preset = cfg.get("sensitivity", "normal")
    return SENSITIVITY_PRESETS.get(preset, SENSITIVITY_PRESETS["normal"])["motion_area_threshold"]