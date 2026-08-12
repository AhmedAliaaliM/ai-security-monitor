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
import threading
import cv2

from core.motion_router import MotionRouter
from core.schedule import get_current_mode
from config.camera_config import get_motion_threshold


class CaptureSession:
    def __init__(self, cfg: dict, on_clip_ready=None, on_frame=None, on_hazard_check=None, hazard_check_interval=15.0, router=None):
        """
        on_hazard_check: optional callback(frame) called on a fixed timer,
                  COMPLETELY INDEPENDENT of motion detection or recording
                  state — hazard checking no longer waits for motion.
        hazard_check_interval: seconds between hazard checks (default 15).
        router: optional pre-built MotionRouter to use instead of creating
                  a private one. Pass the SAME instance a caller also uses
                  elsewhere (e.g. for clip scoring) so live edits - zone,
                  ignore_pets, etc. - reach the motion-detection gate in
                  this loop immediately instead of only taking effect on
                  the next restart. If omitted, a private one is built
                  from `cfg` exactly as before (main.py's CLI usage,
                  where nothing changes mid-run, doesn't need to pass this).
        """
        self.cfg = cfg
        self.on_clip_ready = on_clip_ready or (lambda frames, meta: None)
        self.on_frame = on_frame
        self.on_hazard_check = on_hazard_check
        self.hazard_check_interval = hazard_check_interval
        self._last_hazard_check_time = 0.0

        self.motion_threshold = get_motion_threshold(cfg)
        self.cooldown = cfg.get("cooldown_seconds", 2.5)
        self.max_clip_seconds = cfg.get("max_clip_seconds", 45)
        self.pre_buffer_seconds = cfg.get("pre_buffer_seconds", 1.5)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        self.router = router if router is not None else MotionRouter(cfg)

        self.pre_buffer = collections.deque()  # (timestamp, frame)
        self.recording = False
        self.clip_frames = []
        self.last_motion_time = None
        self.clip_start_time = None

        self._cap = None
        self._fps_estimate = 15.0  # updated once we start reading real frames
        self._consecutive_read_failures = 0

        # Guards against overlapping hazard-check threads (see run()) -
        # without this, setting hazard_check_interval shorter than the
        # model actually takes to run causes threads to pile up, which
        # starves the capture loop of CPU/GIL time badly enough that
        # cv2 reads start failing back-to-back - that's what was making
        # the feed look like it kept closing and reopening.
        self._hazard_check_lock = threading.Lock()
        self._hazard_check_running = False

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
        fgmask = self.router.apply_detection_zone_mask(fgmask)
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
        # Run in a background thread - this callback runs the identity
        # model (and possibly enhancement), which can take real time.
        # Blocking here would freeze frame capture right as a new clip
        # might need to start, causing exactly the watch->record hang.
        frames_to_process = self.clip_frames
        threading.Thread(
            target=self.on_clip_ready, args=(frames_to_process, meta), daemon=True
        ).start()
        self.recording = False
        self.clip_frames = []

    # ---------- hazard check wrapper ----------
    def _run_hazard_check(self, frame):
        """Runs the hazard-check callback, then always clears the
        in-flight flag - in a `finally` so a callback that raises
        doesn't permanently wedge hazard checking off for the rest of
        the session."""
        try:
            self.on_hazard_check(frame)
        finally:
            with self._hazard_check_lock:
                self._hazard_check_running = False

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
                self._consecutive_read_failures += 1
                # A single dropped read on a local webcam is usually a
                # momentary stall (e.g. under CPU load), not a real
                # disconnect - only release+reopen after several in a row,
                # not every time. That "every time" behavior is what was
                # making the feed look like it kept opening and closing.
                if self._consecutive_read_failures >= 10:
                    if self._cap is not None:
                        self._cap.release()
                    self._cap = None
                    self._consecutive_read_failures = 0
                    time.sleep(2.0)
                else:
                    time.sleep(0.05)
                continue

            self._consecutive_read_failures = 0

            mode = get_current_mode(self.cfg)
            significant_contours, all_contours = self._detect_motion(frame)
            motion_now = len(significant_contours) > 0

            if self.on_hazard_check is not None:
                now = time.time()
                if now - self._last_hazard_check_time >= self.hazard_check_interval:
                    with self._hazard_check_lock:
                        already_running = self._hazard_check_running
                        if not already_running:
                            self._hazard_check_running = True
                    if not already_running:
                        self._last_hazard_check_time = now
                        # Run in a background thread - YOLO inference here can
                        # take hundreds of ms to seconds, and blocking the main
                        # loop for that long freezes frame reading AND the
                        # watch->record transition. Guarded above so a check
                        # slower than hazard_check_interval can never overlap
                        # itself - it self-throttles to whatever the model can
                        # actually keep up with instead of piling up threads.
                        threading.Thread(
                            target=self._run_hazard_check, args=(frame.copy(),), daemon=True
                        ).start()
                    # else: previous check is still in flight - skip this
                    # tick, try again next frame (cheap: just a lock+bool).

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

    def reconnect(self):
        """Force the next loop iteration to reopen the capture device -
        e.g. after cfg['source'] changes. This is one deliberate,
        caller-requested close/reopen, not the involuntary repeated kind
        the hazard-check overlap guard above prevents."""
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    def close(self):
        if self._cap is not None:
            self._cap.release()
