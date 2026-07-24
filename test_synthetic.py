"""
Synthetic validation test — no real camera needed.

Generates frames with:
  - a static background (idle, no motion) for a bit
  - a tall-narrow moving rectangle (simulating a person walking through)
  - a wide orange irregular blob (simulating fire/smoke)

This proves the capture -> motion detect -> pre-buffer -> clip -> router
pipeline works end-to-end before testing on real phone/CCTV footage.
"""

import cv2
import numpy as np
import time

from config.camera_config import DEFAULT_CONFIG
from core.capture import CaptureSession
from main import make_on_clip_ready
from core.motion_router import MotionRouter


class SyntheticCapture:
    """Drop-in replacement for cv2.VideoCapture for testing."""

    def __init__(self, total_frames=120):
        self.frame_idx = 0
        self.total_frames = total_frames
        self.w, self.h = 640, 480

    def isOpened(self):
        return True

    def read(self):
        if self.frame_idx >= self.total_frames:
            return False, None

        frame = np.full((self.h, self.w, 3), 40, dtype=np.uint8)  # dark background

        # Phase 1 (frames 0-15):   idle, nothing moving
        # Phase 2 (frames 15-45):  person-like blob walks across
        # Phase 3 (frames 45-60):  idle gap -> forces cooldown, clip should close
        # Phase 4 (frames 60-95):  fire-like orange blob appears
        # Phase 5 (frames 95-150): idle again
        if 15 <= self.frame_idx < 45:
            x = 50 + (self.frame_idx - 15) * 8
            cv2.rectangle(frame, (x, 150), (x + 70, 350), (200, 200, 200), -1)  # tall-narrow, ~2.9:1 ratio
        elif 60 <= self.frame_idx < 95:
            brightness = 255 if self.frame_idx % 2 == 0 else 190  # simulate flicker
            cv2.ellipse(frame, (400, 250), (60 + (self.frame_idx % 10), 40), 0, 0, 360, (0, 100, brightness), -1)  # orange/BGR, flickering

        self.frame_idx += 1
        time.sleep(1 / 15)  # simulate ~15fps so cooldown timing (wall-clock based) behaves realistically
        return True, frame

    def release(self):
        pass


def run_test():
    cfg = dict(DEFAULT_CONFIG)
    cfg["camera_id"] = "synthetic_test_cam"
    cfg["cooldown_seconds"] = 0.3
    cfg["pre_buffer_seconds"] = 0.5
    cfg["sensitivity"] = "normal"

    router = MotionRouter(cfg)
    session = CaptureSession(cfg, on_clip_ready=make_on_clip_ready(cfg, router))
    session._cap = SyntheticCapture(total_frames=160)
    session._ensure_connected = lambda: None  # bypass real connection logic

    print("Running synthetic pipeline test...\n")
    session.run(max_iterations=155)
    print("\nSynthetic test complete.")


if __name__ == "__main__":
    run_test()
