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
from core.schedule import get_current_mode


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


def make_on_clip_ready(cfg, router: MotionRouter):
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

        # Aggregate scoring across the clip, not just one frame — a real
        # event can appear anywhere within the clip's duration, so we
        # sample frames throughout and keep the strongest signal seen.
        from config.camera_config import get_motion_threshold
        threshold = get_motion_threshold(cfg)

        bg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        best_route = None
        ir_votes = []
        sample_stride = max(1, len(frames) // 20)  # cap at ~20 samples per clip

        for i, f in enumerate(frames):
            fgmask = bg.apply(f)  # must run on every frame to keep the model warm
            if i % sample_stride != 0:
                continue
            fgmask_clean = cv2.medianBlur(fgmask, 5)
            contours, _ = cv2.findContours(fgmask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ir_votes.append(detect_ir_mode(f))  # logging only — not used to gate scoring

            # Don't force is_ir_mode from the whole-frame estimate: a small
            # colorful object in a mostly dark/gray frame still averages out
            # to "looks grayscale" at the frame level. The router already
            # checks color availability per-blob, which is the correct scope.
            route = router.score_clip(contours, f, threshold, is_ir_mode=False)
            if best_route is None or (route["hazard_score"] + route["person_score"]) > (
                best_route["hazard_score"] + best_route["person_score"]
            ):
                best_route = route

        ir_mode = sum(ir_votes) > len(ir_votes) / 2 if ir_votes else False
        decision = {"mode": "ai_mode", **best_route, "ir_mode": bool(ir_mode)}

        log_entry = {**meta, "decision": decision}
        print(json.dumps(log_entry, indent=2))
        print("-" * 50)

    return on_clip_ready


def make_on_frame(cfg):
    """
    Builds the on_frame callback used for the live preview window.
    Draws bounding boxes around detected motion, shows recording status,
    and lets 'q' close the window and stop the program cleanly.
    """
    window_name = f"Preview - {cfg.get('camera_id', 'camera')} (press 'q' to quit)"

    def on_frame(frame, significant_contours, is_recording):
        display = frame.copy()

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
    on_frame = make_on_frame(cfg) if args.preview else None
    session = CaptureSession(cfg, on_clip_ready=make_on_clip_ready(cfg, router), on_frame=on_frame)

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
