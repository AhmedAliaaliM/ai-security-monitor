"""
Background engine for the Streamlit dashboard.

This does NOT reimplement any detection logic - it wraps the exact same
core/ pipeline modules main.py uses (CaptureSession, MotionRouter,
PersonDetector, IdentityPipeline, HazardPipeline, LowLightEnhancer) and
adapts their callbacks to write into a thread-safe SharedState object
instead of print()'ing to a console or drawing to a cv2.imshow() window.
app.py only ever reads from SharedState - it never touches cv2/models.

One MotionRouter instance is shared between the engine (clip scoring in
_on_clip_ready) and the CaptureSession it starts (live motion-zone
gating in CaptureSession._detect_motion) - see start(). That's what
makes zone/sensitivity/ignore-pets edits made while monitoring is
already running take effect on the very next frame instead of only on
the next restart.
"""

import collections
import threading
import time

import cv2
import numpy as np

from core.capture import CaptureSession
from core.motion_router import MotionRouter
from core.schedule import is_within_auth_schedule
from core.identity_pipeline import IdentityPipeline, FaceDatabase
from core.hazard_pipeline import HazardPipeline
from core.enhancement_pipeline import LowLightEnhancer
from core.person_detector import PersonDetector
from config.camera_config import get_motion_threshold, SENSITIVITY_PRESETS

KNOWN_FACES_PATH = "known_faces.json"


def _detect_ir_mode(frame) -> bool:
    """Ported unchanged from main.py."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    p95_sat = np.percentile(hsv[:, :, 1], 95)
    return bool(p95_sat < 25)


class SharedState:
    """Thread-safe box: the capture thread writes, the Streamlit UI reads."""

    def __init__(self, max_alerts=200):
        self._lock = threading.Lock()
        self.latest_frame = None       # np.ndarray (BGR), overlays already drawn - for display
        self.latest_raw_frame = None   # np.ndarray (BGR), NO overlays - safe for enrollment capture
        self.frame_shape = None        # (h, w) of the latest frame, once known
        self.is_recording = False
        self.motion_blob_count = 0
        self.running = False
        self.status_message = "Stopped"
        self.alerts = collections.deque(maxlen=max_alerts)  # newest first

    def snapshot(self):
        with self._lock:
            frame = None if self.latest_frame is None else self.latest_frame.copy()
            raw_frame = None if self.latest_raw_frame is None else self.latest_raw_frame.copy()
            return {
                "frame": frame,
                "raw_frame": raw_frame,
                "frame_shape": self.frame_shape,
                "is_recording": self.is_recording,
                "motion_blob_count": self.motion_blob_count,
                "running": self.running,
                "status_message": self.status_message,
                "alerts": list(self.alerts),
            }

    def set_frame(self, frame, raw_frame, is_recording, motion_blob_count):
        with self._lock:
            self.latest_frame = frame
            self.latest_raw_frame = raw_frame
            self.frame_shape = frame.shape[:2]
            self.is_recording = is_recording
            self.motion_blob_count = motion_blob_count

    def push_alert(self, alert: dict):
        with self._lock:
            self.alerts.appendleft(alert)

    def set_status(self, running: bool, message: str):
        with self._lock:
            self.running = running
            self.status_message = message


class SecurityEngine:
    """
    Owns model loading and the CaptureSession background thread.
    Create ONE instance per process - see get_engine() in app.py, cached
    with st.cache_resource so it survives Streamlit reruns and browser
    refreshes instead of orphaning the camera thread.
    """

    def __init__(self, cfg: dict, state: SharedState):
        self.cfg = cfg
        self.state = state

        self.router = None
        self.person_detector = None
        self.identity_pipeline = None
        self.hazard_pipeline = None
        self.enhancer = None
        self.models_loaded = False

        self._session = None
        self._thread = None
        self._stop_flag = threading.Event()

        self._detection_zones = [np.array(z, dtype=np.int32) for z in cfg.get("detection_zones", [])]
        self._ignore_zones = [np.array(z, dtype=np.int32) for z in cfg.get("ignore_zones", [])]

    # ---------------- model loading ----------------

    def load_models(self, on_progress=None):
        if self.models_loaded:
            return

        def report(msg):
            if on_progress:
                on_progress(msg)

        report("Loading person detector (YOLOv8n)...")
        self.person_detector = PersonDetector()
        report("Loading face recognition model...")
        self.identity_pipeline = IdentityPipeline(
            db_path=KNOWN_FACES_PATH,
            match_threshold=self.cfg.get("face_match_threshold", 0.45),
        )
        report("Loading hazard detection model...")
        self.hazard_pipeline = HazardPipeline(
            confidence_threshold=self.cfg.get("hazard_confidence_threshold", 0.2),
        )
        report("Loading low-light enhancement model...")
        self.enhancer = LowLightEnhancer()
        self.router = MotionRouter(self.cfg)
        self.models_loaded = True

    # ---------------- live config edits (no restart needed) ----------------
    # Every method below updates self.cfg (the single source of truth - a
    # fresh load_models()/start() would read exactly this) AND, if a
    # router/session already exists, pushes the same change directly onto
    # it so an edit made while monitoring is already running takes effect
    # on the very next frame instead of waiting for a restart.

    def update_detection_zone(self, rect):
        """rect: (x1, y1, x2, y2) in frame pixel coords. Applies immediately."""
        x1, y1, x2, y2 = rect
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        zones_np = [np.array(polygon, dtype=np.int32)]

        self.cfg["detection_zones"] = [polygon]
        self._detection_zones = zones_np
        if self.router is not None:
            self.router.detection_zones = zones_np
            self.router._zone_mask = None  # force the cached pixel mask to rebuild

    def update_ignore_zone(self, rect):
        """rect: (x1, y1, x2, y2) in frame pixel coords, or None to clear it."""
        if rect is None:
            self.cfg["ignore_zones"] = []
            zones_np = []
        else:
            x1, y1, x2, y2 = rect
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            self.cfg["ignore_zones"] = [polygon]
            zones_np = [np.array(polygon, dtype=np.int32)]

        self._ignore_zones = zones_np
        if self.router is not None:
            self.router.ignore_zones = zones_np
            # apply_ignore_mask() doesn't cache a mask - nothing else to invalidate

    def update_sensitivity(self, preset: str):
        """Quick-set: applies a preset's motion_area_threshold. The number
        itself (update_motion_threshold) can still be fine-tuned afterward -
        this just gives it a sane starting point."""
        self.cfg["sensitivity"] = preset
        if preset in SENSITIVITY_PRESETS:
            self.cfg["motion_area_threshold"] = SENSITIVITY_PRESETS[preset]["motion_area_threshold"]
        if self._session is not None:
            self._session.motion_threshold = get_motion_threshold(self.cfg)

    def update_motion_threshold(self, value: int):
        """Direct fine-tuning of the actual pixel-area cutoff, independent
        of which preset (if any) it currently matches."""
        self.cfg["motion_area_threshold"] = value
        if self._session is not None:
            self._session.motion_threshold = get_motion_threshold(self.cfg)

    def update_hazard_confidence_threshold(self, value: float):
        self.cfg["hazard_confidence_threshold"] = value
        if self.hazard_pipeline is not None:
            self.hazard_pipeline.confidence_threshold = value

    def update_face_match_threshold(self, value: float):
        self.cfg["face_match_threshold"] = value
        if self.identity_pipeline is not None:
            self.identity_pipeline.match_threshold = value

    def update_person_threshold(self, value: float):
        self.cfg["person_confidence_threshold"] = value
        if self.router is not None:
            self.router.person_threshold = value

    def update_ignore_pets(self, value: bool):
        self.cfg["ignore_pets"] = bool(value)
        if self.router is not None:
            self.router.ignore_pets = bool(value)

    def update_recording_params(self, cooldown_seconds=None, max_clip_seconds=None, pre_buffer_seconds=None):
        if cooldown_seconds is not None:
            self.cfg["cooldown_seconds"] = cooldown_seconds
            if self._session is not None:
                self._session.cooldown = cooldown_seconds
        if max_clip_seconds is not None:
            self.cfg["max_clip_seconds"] = max_clip_seconds
            if self._session is not None:
                self._session.max_clip_seconds = max_clip_seconds
        if pre_buffer_seconds is not None:
            self.cfg["pre_buffer_seconds"] = pre_buffer_seconds
            if self._session is not None:
                self._session.pre_buffer_seconds = pre_buffer_seconds

    def update_repetitive_motion(self, window=None, max_triggers=None):
        if window is not None:
            self.cfg["repetitive_motion_window"] = window
            if self.router is not None:
                self.router.window = window
        if max_triggers is not None:
            self.cfg["repetitive_motion_max_triggers"] = max_triggers
            if self.router is not None:
                self.router.max_triggers = max_triggers

    def update_hazard_interval(self, seconds: float):
        self.cfg["hazard_check_interval"] = seconds
        if self._session is not None:
            self._session.hazard_check_interval = seconds

    def update_schedule_ai_mode(self, windows):
        """windows: list of [start_hhmm, end_hhmm] pairs. get_current_mode()
        reads self.cfg fresh on every call, so this needs no push to the
        running session - it applies on the very next frame."""
        self.cfg["schedule"] = {"ai_mode": windows}

    def update_auth_schedule(self, days, start, end):
        """is_within_auth_schedule() reads self.cfg fresh on every call,
        same as above - no push needed."""
        self.cfg["auth_schedule"] = {"days": days, "start": start, "end": end}

    def update_source(self, source):
        """Forces exactly one deliberate reconnect if already running -
        the camera briefly closes/reopens ONCE, which is expected here
        (you just told it to point somewhere else) - unlike the repeated
        involuntary kind the hazard-check overlap guard prevents."""
        self.cfg["source"] = source
        if self._session is not None:
            self._session.reconnect()

    # ---------------- known faces ----------------

    def list_known_faces(self):
        """Names currently enrolled. Works even before models are loaded -
        this is just a JSON read, no ML needed for a listing."""
        if self.identity_pipeline is not None:
            return sorted(self.identity_pipeline.db.entries.keys())
        return sorted(FaceDatabase(KNOWN_FACES_PATH).entries.keys())

    def remove_known_face(self, name: str):
        if self.identity_pipeline is not None:
            self.identity_pipeline.db.remove(name)
        else:
            FaceDatabase(KNOWN_FACES_PATH).remove(name)

    def enroll_face(self, name: str, frame_bgr):
        """Loads models first if needed (same one-time cost as Start
        monitoring - load_models() is a no-op if already loaded), so
        enrollment always writes into the SAME FaceDatabase object the
        live detection loop reads, never a second out-of-sync copy.
        Returns True if a face was found in the image and enrolled."""
        if not self.models_loaded:
            self.load_models()
        return self.identity_pipeline.enroll_from_image(name, frame_bgr)

    # ---------------- callbacks (ported from main.py -> write to SharedState) ----------------

    def _on_clip_ready(self, frames, meta):
        if not frames:
            return
        mode = meta["mode"]

        if mode == "cctv_mode":
            self.state.push_alert({
                "type": "clip",
                **meta,
                "decision": {"mode": "cctv_mode", "action": "saved_only_no_ai"},
            })
            return

        threshold = get_motion_threshold(self.cfg)
        bg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        best_route = None
        best_person_frame = None
        best_person_score = -1
        ir_votes = []
        sample_stride = max(1, len(frames) // 20)

        for i, f in enumerate(frames):
            fgmask = bg.apply(f)
            if i % sample_stride != 0:
                continue
            fgmask_clean = cv2.medianBlur(fgmask, 5)
            contours, _ = cv2.findContours(fgmask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ir_votes.append(_detect_ir_mode(f))

            route = self.router.score_clip(contours, f, threshold, is_ir_mode=False, person_detector=self.person_detector)
            if best_route is None or route["person_score"] > best_route["person_score"]:
                best_route = route
            if route["person_score"] > best_person_score:
                best_person_score = route["person_score"]
                best_person_frame = f

        ir_mode = sum(ir_votes) > len(ir_votes) / 2 if ir_votes else False
        decision = {"mode": "ai_mode", **best_route, "ir_mode": bool(ir_mode)}

        identity_decision = None
        if best_route.get("run_pipeline_b") and best_person_frame is not None:
            if is_within_auth_schedule(self.cfg):
                person_input_frame = best_person_frame
                enhanced_for_identity = False
                if self.enhancer is not None and self.enhancer.needs_enhancement(person_input_frame):
                    person_input_frame = self.enhancer.enhance(person_input_frame)
                    enhanced_for_identity = True
                face_results = self.identity_pipeline.process_frame(person_input_frame)
                identity_decision = self.identity_pipeline.decide(face_results)
                identity_decision["enhanced"] = enhanced_for_identity
                decision["identity"] = identity_decision
            else:
                identity_decision = {
                    "status": "UNAUTHORIZED",
                    "detail": None,
                    "reason": "outside_auth_schedule",
                    "enhanced": False,
                }
                decision["identity"] = identity_decision

        if identity_decision and identity_decision["status"] == "authorized":
            final_action = "SAFE_ENTRY"
        elif identity_decision and identity_decision["status"] == "UNAUTHORIZED":
            final_action = "UNAUTHORIZED_ALERT_UNKNOWN_FACE"
        else:
            final_action = "LOG_ONLY"
        decision["final_action"] = final_action

        self.state.push_alert({"type": "clip", **meta, "decision": decision})

    def _on_hazard_check(self, frame):
        input_frame = frame
        enhanced = False
        if self.enhancer is not None and self.enhancer.needs_enhancement(input_frame):
            input_frame = self.enhancer.enhance(input_frame)
            enhanced = True

        detections = self.hazard_pipeline.process_frame(input_frame)
        decision = self.hazard_pipeline.decide(detections)
        decision["enhanced"] = enhanced

        if decision["status"] == "HAZARD":
            self.state.push_alert({
                "type": "hazard",
                "timestamp": time.time(),
                "decision": decision,
                "final_action": "UNAUTHORIZED_ALERT_HAZARD",
            })
        # no_hazard checks are intentionally not pushed - the alerts feed
        # is for things that actually need attention.

    def _on_frame(self, frame, significant_contours, is_recording):
        display = frame.copy()

        if self._detection_zones:
            overlay = display.copy()
            for poly in self._detection_zones:
                cv2.fillPoly(overlay, [poly], (255, 0, 0))
            display = cv2.addWeighted(overlay, 0.15, display, 0.85, 0)
            for poly in self._detection_zones:
                cv2.polylines(display, [poly], isClosed=True, color=(255, 0, 0), thickness=2)

        if self._ignore_zones:
            for poly in self._ignore_zones:
                cv2.polylines(display, [poly], isClosed=True, color=(150, 150, 150), thickness=2)

        for c in significant_contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # raw `frame` (no overlays) is stashed separately so face enrollment
        # can grab a clean shot without a zone box potentially drawn over
        # someone's face, and without ever opening a second cv2.VideoCapture.
        self.state.set_frame(display, frame, is_recording, len(significant_contours))
        return not self._stop_flag.is_set()

    # ---------------- lifecycle ----------------

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.models_loaded:
            raise RuntimeError("Call load_models() before start().")

        self._stop_flag.clear()
        self._session = CaptureSession(
            self.cfg,
            on_clip_ready=self._on_clip_ready,
            on_frame=self._on_frame,
            on_hazard_check=self._on_hazard_check,
            hazard_check_interval=self.cfg.get("hazard_check_interval", 2.0),
            router=self.router,
        )
        self.state.set_status(True, "Running")

        def _run():
            try:
                self._session.run()
            finally:
                self._session.close()
                self.state.set_status(False, "Stopped")

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None or not self._thread.is_alive():
            return
        self._stop_flag.set()
        self.state.set_status(True, "Stopping...")