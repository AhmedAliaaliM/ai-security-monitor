"""
Step 1 entry point: motion detection + heuristic router, validated locally,
console output only. No cloud, no heavy AI models yet — this just proves
the capture/trigger/routing logic works before we wire in Zero-DCE++,
YOLO, and face recognition.

Usage:
    python main.py                      # uses default webcam (source=0)
    python main.py --source <rtsp_url>  # for CCTV/DroidCam testing
    python main.py --config path.json   # load a saved camera config
"""

import argparse
import json
import time
import sys

import cv2
import numpy as np

from config.camera_config import load_camera_config, save_default_config, DEFAULT_CONFIG
from core.capture import CaptureSession
from core.motion_router import MotionRouter
from core.schedule import get_current_mode, is_within_auth_schedule
from core.identity_pipeline import IdentityPipeline
from core.hazard_pipeline import HazardPipeline
from core.enhancement_pipeline import LowLightEnhancer
from core.person_detector import PersonDetector


def detect_ir_mode(frame) -> bool:
    """
    Cheap heuristic: IR/night-mode footage is near-grayscale EVERYWHERE,
    including any bright/moving objects in frame. Using the mean saturation
    across the whole frame is misleading — a small colorful object against
    a large dark background still averages out low. Instead check the 95th
    percentile, which reflects the most saturated real content in the frame.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    p95_sat = np.percentile(hsv[:, :, 1], 95)
    return bool(p95_sat < 25)


def make_on_clip_ready(cfg, router: MotionRouter, identity_pipeline: IdentityPipeline = None, enhancer: LowLightEnhancer = None, person_detector: PersonDetector = None):
    """
    Handles the IDENTITY/PERSON side only now. Hazard detection has moved
    to its own independent timer (see make_on_hazard_check) — it no longer
    goes through motion detection or any heuristic gate at all.
    """
    def on_clip_ready(frames, meta):
        if not frames:
            print(f"[{meta['camera_id']}] Empty clip, skipping.")
            return

        mode = meta["mode"]

        if mode == "cctv_mode":
            decision = {"mode": "cctv_mode", "action": "saved_only_no_ai"}
            log_entry = {**meta, "decision": decision}
            print(json.dumps(log_entry, indent=2))
            print("-" * 50)
            return

        from config.camera_config import get_motion_threshold
        threshold = get_motion_threshold(cfg)

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
            ir_votes.append(detect_ir_mode(f))

            route = router.score_clip(contours, f, threshold, is_ir_mode=False, person_detector=person_detector)
            if best_route is None or route["person_score"] > best_route["person_score"]:
                best_route = route

            if route["person_score"] > best_person_score:
                best_person_score = route["person_score"]
                best_person_frame = f

        ir_mode = sum(ir_votes) > len(ir_votes) / 2 if ir_votes else False
        decision = {"mode": "ai_mode", **best_route, "ir_mode": bool(ir_mode)}

        identity_decision = None
        if best_route.get("run_pipeline_b") and best_person_frame is not None:
            if is_within_auth_schedule(cfg):
                person_input_frame = best_person_frame
                enhanced_for_identity = False
                if enhancer is not None and enhancer.needs_enhancement(person_input_frame):
                    person_input_frame = enhancer.enhance(person_input_frame)
                    enhanced_for_identity = True
                if identity_pipeline is not None:
                    face_results = identity_pipeline.process_frame(person_input_frame)
                    identity_decision = identity_pipeline.decide(face_results)
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

        log_entry = {**meta, "decision": decision}
        print(json.dumps(log_entry, indent=2))
        print("-" * 50)

    return on_clip_ready


def make_on_hazard_check(hazard_pipeline: HazardPipeline, enhancer: LowLightEnhancer = None):
    """
    Runs the real fire/smoke model on a fixed timer (see CaptureSession's
    hazard_check_interval), completely independent of motion detection.
    No heuristic gate at all — just "has it been N seconds, then check".
    """
    def on_hazard_check(frame):
        input_frame = frame
        enhanced = False
        if enhancer is not None and enhancer.needs_enhancement(input_frame):
            input_frame = enhancer.enhance(input_frame)
            enhanced = True

        detections = hazard_pipeline.process_frame(input_frame)
        decision = hazard_pipeline.decide(detections)
        decision["enhanced"] = enhanced

        

        log_entry = {
            "check_type": "periodic_hazard_check",
            "timestamp": time.time(),
            "decision": decision,
        }
        if decision["status"] == "HAZARD":
            log_entry["final_action"] = "UNAUTHORIZED_ALERT_HAZARD"
            print(json.dumps(log_entry, indent=2))
            print("-" * 50)
        # Non-hazard results are intentionally not printed to keep the
        # console readable — every 15s would otherwise flood the log with
        # "no_hazard" lines. Change this if you want to see every check.

    return on_hazard_check


def make_on_frame(cfg):
    """
    Builds the on_frame callback used for the live preview window.
    Draws bounding boxes around detected motion, shows recording status,
    draws configured detection_zones (blue) and ignore_zones (gray) so
    you can visually confirm where they actually are, and lets 'q' close
    the window and stop the program cleanly.
    """
    window_name = f"Preview - {cfg.get('camera_id', 'camera')} (press 'q' to quit)"

    detection_zones = [np.array(z, dtype=np.int32) for z in cfg.get("detection_zones", [])]
    ignore_zones = [np.array(z, dtype=np.int32) for z in cfg.get("ignore_zones", [])]

    def on_frame(frame, significant_contours, is_recording):
        display = frame.copy()

        if detection_zones:
            overlay = display.copy()
            for poly in detection_zones:
                cv2.fillPoly(overlay, [poly], (255, 0, 0))
            display = cv2.addWeighted(overlay, 0.15, display, 0.85, 0)
            for poly in detection_zones:
                cv2.polylines(display, [poly], isClosed=True, color=(255, 0, 0), thickness=2)
            label_pos = tuple(detection_zones[0][0])
            cv2.putText(display, "DETECTION ZONE", (label_pos[0], max(20, label_pos[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if ignore_zones:
            for poly in ignore_zones:
                cv2.polylines(display, [poly], isClosed=True, color=(150, 150, 150), thickness=2)

        for c in significant_contours:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

        status_text = "RECORDING" if is_recording else "watching..."
        status_color = (0, 0, 255) if is_recording else (200, 200, 200)
        cv2.putText(display, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, status_color, 2)
        cv2.putText(display, f"motion blobs: {len(significant_contours)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(window_name, display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return False  # tells CaptureSession.run() to stop the loop
        return True

    return on_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None, help="Webcam index, RTSP URL, or DroidCam URL")
    parser.add_argument("--config", type=str, default=None, help="Path to a camera config JSON file")
    parser.add_argument("--init-config", type=str, default=None, help="Write a starter config file to this path and exit")
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop after N frames (testing only)")
    parser.add_argument("--preview", action="store_true", help="Show a live window with motion boxes (press 'q' to quit)")
    args = parser.parse_args()

    if args.init_config:
        save_default_config(args.init_config)
        print(f"Starter config written to {args.init_config}")
        return

    cfg = load_camera_config(args.config) if args.config else dict(DEFAULT_CONFIG)
    if args.source is not None:
        try:
            cfg["source"] = int(args.source)
        except ValueError:
            cfg["source"] = args.source

    print(f"Starting capture for camera_id='{cfg['camera_id']}' source={cfg['source']}")
    print(f"Current mode: {get_current_mode(cfg)}")
    if args.preview:
        print("Preview window enabled — click the window and press 'q' to quit.")

    router = MotionRouter(cfg)
    print("Loading person detector (base YOLOv8n, COCO)...")
    person_detector = PersonDetector()
    print("Loading face recognition model (one-time, may take a few seconds)...")
    identity_pipeline = IdentityPipeline(match_threshold=cfg.get("face_match_threshold", 0.45))
    print("Loading hazard detection model (one-time, may take a few seconds)...")
    hazard_pipeline = HazardPipeline(confidence_threshold=cfg.get("hazard_confidence_threshold", 0.2))
    print("Loading low-light enhancement model (one-time, may take a few seconds)...")
    enhancer = LowLightEnhancer()
    on_frame = make_on_frame(cfg) if args.preview else None
    session = CaptureSession(
        cfg,
        on_clip_ready=make_on_clip_ready(cfg, router, identity_pipeline, enhancer, person_detector),
        on_frame=on_frame,
        on_hazard_check=make_on_hazard_check(hazard_pipeline, enhancer),
        hazard_check_interval=cfg.get("hazard_check_interval", 2.0),
        router=router,
    )

    try:
        session.run(max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        session.close()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()