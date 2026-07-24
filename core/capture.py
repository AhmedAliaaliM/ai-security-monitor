"""
Capture loop.

- Continuously reads frames and runs background subtraction (cheap, always on).
- Maintains a rolling pre-buffer so a triggered clip includes a moment
  BEFORE the trigger, not just after.
- On motion: starts a variable-length clip, keeps recording while motion
  continues, stops after `cooldown_seconds` of no motion or `max_clip_seconds`
  hard cap — whichever comes first.
- Handles RTSP reconnect so a dropped stream doesn't kill the process.
"""

import time
import collections
import cv2

from core.motion_router import MotionRouter
from core.schedule import get_current_mode
from config.camera_config import get_motion_threshold


class CaptureSession:
    def __init__(self, cfg: dict, on_clip_ready=None, on_frame=None):
        """
        cfg: camera config dict (see config/camera_config.py)
        on_clip_ready: callback(frames: list[np.ndarray], meta: dict) called
                       once a clip finishes recording.
        on_frame: optional callback(frame, significant_contours, is_recording) called
                  every single frame — used for live preview/debugging. Should
                  return False to request the loop stop (e.g. user pressed 'q'),
                  anything else (including None) continues normally.
        """
        self.cfg = cfg
        self.on_clip_ready = on_clip_ready or (lambda frames, meta: None)
        self.on_frame = on_frame

        self.motion_threshold = get_motion_threshold(cfg)
        self.cooldown = cfg.get("cooldown_seconds", 2.5)
        self.max_clip_seconds = cfg.get("max_clip_seconds", 45)
        self.pre_buffer_seconds = cfg.get("pre_buffer_seconds", 1.5)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.router = MotionRouter(cfg)

        self.pre_buffer = collections.deque()  # (timestamp, frame)
        self.recording = False
        self.clip_frames = []
        self.last_motion_time = None
        self.clip_start_time = None

        self._cap = None
        self._fps_estimate = 15.0  # updated once we start reading real frames

    # ---------- connection handling ----------
    def _open_capture(self):
        source = self.cfg.get("source", 0)
        cap = cv2.VideoCapture(source)
        return cap

    def _ensure_connected(self):
        if self._cap is None or not self._cap.isOpened():
            if self._cap is not None:
                self._cap.release()
            self._cap = self._open_capture()
            time.sleep(1.0)

    # ---------- pre-buffer management ----------
    def _push_prebuffer(self, frame):
        now = time.time()
        self.pre_buffer.append((now, frame))
        while self.pre_buffer and now - self.pre_buffer[0][0] > self.pre_buffer_seconds:
            self.pre_buffer.popleft()

    def _prebuffer_frames(self):
        return [f for _, f in self.pre_buffer]

    # ---------- motion detection ----------
    def _detect_motion(self, frame):
        fgmask = self.bg_subtractor.apply(frame)
        fgmask = cv2.medianBlur(fgmask, 5)
        fgmask = self.router.apply_ignore_mask(fgmask)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant = [c for c in contours if cv2.contourArea(c) >= self.motion_threshold]
        return significant, contours

    # ---------- clip lifecycle ----------
    def _start_clip(self):
        self.recording = True
        self.clip_frames = self._prebuffer_frames()  # seed with pre-trigger frames
        self.clip_start_time = time.time()
        self.last_motion_time = time.time()

    def _finish_clip(self, mode: str):
        duration = time.time() - self.clip_start_time
        meta = {
            "camera_id": self.cfg.get("camera_id"),
            "duration_seconds": round(duration, 2),
            "frame_count": len(self.clip_frames),
            "mode": mode,  # 'ai_mode' or 'cctv_mode'
            "timestamp": self.clip_start_time,
        }
        self.on_clip_ready(self.clip_frames, meta)
        self.recording = False
        self.clip_frames = []

    # ---------- main loop ----------
    def run(self, max_iterations=None):
        """
        max_iterations: for testing only, stops the loop after N frames.
        In production this runs forever (or until the process is killed).
        """
        self._ensure_connected()
        iterations = 0

        while True:
            self._ensure_connected()
            ret, frame = self._cap.read()

            if not ret:
                # stream dropped — release and retry
                if self._cap is not None:
                    self._cap.release()
                self._cap = None
                time.sleep(2.0)
                continue

            mode = get_current_mode(self.cfg)
            significant_contours, all_contours = self._detect_motion(frame)
            motion_now = len(significant_contours) > 0

            if not self.recording:
                self._push_prebuffer(frame)
                if motion_now:
                    self._start_clip()
                    self.clip_frames.append(frame)
            else:
                self.clip_frames.append(frame)
                if motion_now:
                    self.last_motion_time = time.time()

                no_motion_duration = time.time() - self.last_motion_time
                clip_duration = time.time() - self.clip_start_time

                if no_motion_duration > self.cooldown:
                    self._finish_clip(mode)
                elif clip_duration > self.max_clip_seconds:
                    self._finish_clip(mode)

            if self.on_frame is not None:
                should_continue = self.on_frame(frame, significant_contours, self.recording)
                if should_continue is False:
                    break

            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break

    def close(self):
        if self._cap is not None:
            self._cap.release()
