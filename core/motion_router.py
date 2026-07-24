"""
Heuristic router (Option 2): decides which heavy AI pipeline(s) to run,
without using any neural network. Cheap feature extraction on motion
contours only.

Outputs:
    - hazard_score, person_score (0.0 - 1.0-ish, not calibrated probabilities)
    - run_pipeline_a (hazard), run_pipeline_b (identity) booleans
    - is_pet, is_ignored_zone, is_repetitive flags for transparency/logging
"""

import time
import cv2
import numpy as np

HAZARD_THRESHOLD = 0.5
PERSON_THRESHOLD = 0.5


class MotionRouter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ignore_zones = [np.array(z, dtype=np.int32) for z in cfg.get("ignore_zones", [])]
        self.ignore_pets = cfg.get("ignore_pets", False)
        self.motion_history = {}  # region_key -> list of timestamps
        self.window = cfg.get("repetitive_motion_window", 60)
        self.max_triggers = cfg.get("repetitive_motion_max_triggers", 5)
        self.brightness_history = {}  # region_key -> list of (timestamp, mean_value)
        self.flicker_window = 3.0  # seconds of history used to judge flicker

    # ---------- zone masking ----------
    def apply_ignore_mask(self, fgmask: np.ndarray) -> np.ndarray:
        if not self.ignore_zones:
            return fgmask
        masked = fgmask.copy()
        for poly in self.ignore_zones:
            cv2.fillPoly(masked, [poly], 0)
        return masked

    # ---------- repetitive motion (same spot triggering constantly) ----------
    def _region_key(self, bbox, grid_size=8, frame_shape=(480, 640)):
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        gx = int(cx / frame_shape[1] * grid_size)
        gy = int(cy / frame_shape[0] * grid_size)
        return (gx, gy)

    def _is_repetitive(self, bbox, frame_shape) -> bool:
        key = self._region_key(bbox, frame_shape=frame_shape)
        now = time.time()
        history = [t for t in self.motion_history.get(key, []) if now - t < self.window]
        history.append(now)
        self.motion_history[key] = history
        return len(history) > self.max_triggers

    # ---------- flicker tracking (fire flickers, skin/faces don't) ----------
    def _flicker_score(self, bbox, frame_shape, mean_value) -> float:
        key = self._region_key(bbox, frame_shape=frame_shape)
        now = time.time()
        history = [(t, v) for t, v in self.brightness_history.get(key, []) if now - t < self.flicker_window]
        history.append((now, mean_value))
        self.brightness_history[key] = history

        if len(history) < 3:
            return 0.0  # not enough samples yet to judge flicker — assume none

        values = np.array([v for _, v in history])
        std = float(np.std(values))
        # normalize: real flame flicker typically swings brightness by 15-40+
        # units on a 0-255 scale between frames; static warm objects (skin,
        # painted walls, furniture) barely move.
        return min(1.0, std / 25.0)

    # ---------- pet-shape heuristic ----------
    def _is_likely_pet(self, bbox, frame_height) -> bool:
        x, y, w, h = bbox
        if w == 0:
            return False
        aspect_ratio = h / w
        relative_height = h / frame_height
        return relative_height < 0.35 and aspect_ratio < 1.2

    # ---------- feature extraction per contour ----------
    def _contour_features(self, contour, frame_hsv, is_ir_mode=False):
        """
        is_ir_mode here is a per-blob decision based on that blob's OWN
        region — not a whole-frame flag. A frame-wide saturation check is
        misleading whenever the object of interest is small relative to
        the frame (e.g. a small fire in a large dark scene averages out
        to "looks grayscale" even though the object itself is vividly
        colored). Checking saturation within just the blob's bounding box
        is the correct scope for this decision.
        """
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 1
        solidity = area / hull_area
        aspect_ratio = h / w if w > 0 else 0

        # Build a mask of ONLY the actual contour shape, not its bounding
        # rectangle. A bounding-box crop includes a lot of surrounding
        # pixels that aren't part of the blob at all (e.g. a hand holding
        # a small lighter flame) — scoring the whole rectangle dilutes a
        # small bright flame into near-nothing. Scoring only the real
        # shape's pixels keeps the signal intact.
        blob_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour - [x, y]
        cv2.drawContours(blob_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

        roi = None
        roi_pixels = None
        if frame_hsv is not None:
            roi = frame_hsv[y:y + h, x:x + w]
            if roi.size > 0:
                roi_pixels = roi[blob_mask > 0]  # only real contour pixels, (N, 3)

        blob_is_grayscale = is_ir_mode
        if roi_pixels is not None and roi_pixels.size > 0 and not is_ir_mode:
            blob_is_grayscale = np.percentile(roi_pixels[:, 1], 90) < 25

        hazard_color_score = 0.0
        if not blob_is_grayscale and roi is not None and roi.size > 0:
            hue2d = roi[:, :, 0]
            sat2d = roi[:, :, 1]
            val2d = roi[:, :, 2]
            fire_mask_2d = (((hue2d < 25) | (hue2d > 168)) & (sat2d > 130) & (val2d > 180)).astype(np.uint8)
            fire_mask_2d = fire_mask_2d & (blob_mask > 0)  # only within the actual blob shape

            # KEY FIX: don't average over the whole blob — a small flame
            # inside a much larger hand/arm blob would always look tiny as
            # a fraction, no matter how bright the flame itself is. Instead,
            # find the largest CONNECTED cluster of fire-colored pixels and
            # judge it by absolute pixel count. A real flame forms a solid
            # clump of matching pixels; scattered single-pixel noise won't.
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fire_mask_2d, connectivity=8)
            largest_cluster_area = 0
            largest_cluster_mask = None
            if num_labels > 1:  # label 0 is background
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_idx = int(np.argmax(areas)) + 1
                largest_cluster_area = int(areas[largest_idx - 1])
                largest_cluster_mask = (labels == largest_idx)

            MIN_CLUSTER_PIXELS = 40  # absolute size floor — tune per camera resolution
            if largest_cluster_area >= MIN_CLUSTER_PIXELS and largest_cluster_mask is not None:
                # Confidence scales with cluster size (capped) rather than
                # blob-relative fraction.
                color_score = min(1.0, largest_cluster_area / 300.0)

                # Flicker measured ONLY on the flame-colored sub-region's
                # brightness, not the whole blob (whose brightness is mostly
                # determined by the much larger, non-flickering hand/arm).
                mean_value = float(np.mean(val2d[largest_cluster_mask]))
                flicker = self._flicker_score(
                    (x, y, w, h), frame_hsv.shape[:2] if frame_hsv is not None else (480, 640), mean_value
                )
                hazard_color_score = color_score * (0.3 + 0.7 * flicker)
        elif blob_is_grayscale:
            # No usable color signal for this blob: fall back to a weak
            # texture proxy (irregular/low-solidity shapes read as more
            # smoke/flame-like than rigid ones).
            hazard_color_score = max(0.0, 1.0 - solidity) * 0.5

        return {
            "bbox": (x, y, w, h),
            "area": area,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "hazard_color_score": hazard_color_score,
        }

    # ---------- main entry point ----------
    def score_clip(self, contours, frame_bgr, motion_area_threshold, is_ir_mode=False):
        """
        contours: list of cv2 contours from the motion mask for one frame
        frame_bgr: the current frame (for color histogram features)
        Returns a decision dict.
        """
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV) if not is_ir_mode else None

        hazard_score = 0.0
        person_score = 0.0
        pet_detected = False
        repetitive_detected = False
        valid_blob_found = False

        for c in contours:
            area = cv2.contourArea(c)
            if area < motion_area_threshold:
                continue  # too small given current sensitivity preset
            valid_blob_found = True

            feats = self._contour_features(c, frame_hsv, is_ir_mode=is_ir_mode)
            bbox = feats["bbox"]

            if self.ignore_pets and self._is_likely_pet(bbox, frame_h):
                pet_detected = True
                continue

            # person-likelihood: tall-narrow, higher solidity (rigid shape)
            aspect_ok = 1.5 <= feats["aspect_ratio"] <= 4.0
            solidity_ok = feats["solidity"] > 0.5
            candidate_person_score = 0.6 + 0.4 * feats["solidity"] if (aspect_ok and solidity_ok) else 0.0
            candidate_hazard_score = feats["hazard_color_score"]

            # Repetitive-motion suppression is meant for environmental noise
            # (curtains, fans, tree branches) — NOT for a real hazard or
            # person that happens to stay in the same region (e.g. a fire
            # that hasn't spread, or someone standing still). Only suppress
            # when there's no meaningful signal underneath; a real hazard or
            # person score always overrides the repetitive flag.
            repetitive = self._is_repetitive(bbox, (frame_h, frame_w))
            has_real_signal = candidate_hazard_score > 0.15 or candidate_person_score > 0
            if repetitive and not has_real_signal:
                repetitive_detected = True
                continue

            person_score = max(person_score, candidate_person_score)
            hazard_score = max(hazard_score, candidate_hazard_score)

        return {
            "hazard_score": round(float(hazard_score), 3),
            "person_score": round(float(person_score), 3),
            "run_pipeline_a": bool(hazard_score > HAZARD_THRESHOLD),
            "run_pipeline_b": bool(person_score > PERSON_THRESHOLD),
            "is_pet_filtered": bool(pet_detected),
            "is_repetitive_filtered": bool(repetitive_detected),
            "valid_blob_found": bool(valid_blob_found),
        }
