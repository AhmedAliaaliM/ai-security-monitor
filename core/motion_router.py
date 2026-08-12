"""
Router: decides whether the person/identity pipeline should run.
Hazard detection is NOT handled here anymore - HazardPipeline runs
independently on its own timer (see main.py). Person scoring comes
from base YOLOv8n (COCO) when provided - more diverse/reliable than
the custom classifier's person output on arbitrary webcam footage,
and gated to actual detected-person boxes inside detection_zones
(not just nearby motion blobs). Pet-filtering, ignore-zones, and
repetitive-motion suppression are still hardcoded shape/location
heuristics, independent of either model.
"""

import time
import cv2
import numpy as np

PERSON_THRESHOLD = 0.4


class MotionRouter:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.ignore_zones = [np.array(z, dtype=np.int32) for z in cfg.get("ignore_zones", [])]
        self.detection_zones = [np.array(z, dtype=np.int32) for z in cfg.get("detection_zones", [])]
        self.ignore_pets = cfg.get("ignore_pets", False)
        self.motion_history = {}
        self.window = cfg.get("repetitive_motion_window", 60)
        self.max_triggers = cfg.get("repetitive_motion_max_triggers", 5)
        self.person_threshold = cfg.get("person_confidence_threshold", PERSON_THRESHOLD)
        self.brightness_history = {}
        self.flicker_window = 3.0
        self._zone_mask = None  # lazily built, cached per frame size

    def apply_ignore_mask(self, fgmask: np.ndarray) -> np.ndarray:
        if not self.ignore_zones:
            return fgmask
        masked = fgmask.copy()
        for poly in self.ignore_zones:
            cv2.fillPoly(masked, [poly], 0)
        return masked

    def _get_zone_mask(self, shape):
        h, w = shape
        if self._zone_mask is None or self._zone_mask.shape != (h, w):
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in self.detection_zones:
                cv2.fillPoly(mask, [poly], 255)
            self._zone_mask = mask
        return self._zone_mask

    def apply_detection_zone_mask(self, fgmask: np.ndarray) -> np.ndarray:
        """Zero out motion pixels OUTSIDE detection_zones."""
        if not self.detection_zones:
            return fgmask
        mask = self._get_zone_mask(fgmask.shape[:2])
        return cv2.bitwise_and(fgmask, mask)


    def _bbox_center_in_detection_zone(self, bbox) -> bool:
        if not self.detection_zones:
            return True
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        for poly in self.detection_zones:
            if cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0:
                return True
        return False

    def _bbox_overlaps_detection_zone(self, bbox, frame_shape) -> bool:
        """
        Unlike _bbox_center_in_detection_zone, this checks whether ANY part
        of the box touches a zone - needed for full-body person boxes, where
        the box CENTER (torso height) can sit well outside a zone drawn
        around e.g. a doorway or a face-height band, even though the top of
        that same box (head/face) is clearly inside it.
        """
        if not self.detection_zones:
            return True
        h, w = frame_shape
        if self._zone_mask is None or self._zone_mask.shape != (h, w):
            mask = np.zeros((h, w), dtype=np.uint8)
            for poly in self.detection_zones:
                cv2.fillPoly(mask, [poly], 255)
            self._zone_mask = mask

        x, y, bw, bh = bbox
        x0, y0 = max(0, int(x)), max(0, int(y))
        x1, y1 = min(w, int(x + bw)), min(h, int(y + bh))
        if x1 <= x0 or y1 <= y0:
            return False
        return bool(np.any(self._zone_mask[y0:y1, x0:x1]))

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

    def _flicker_score(self, bbox, frame_shape, mean_value) -> float:
        key = self._region_key(bbox, frame_shape=frame_shape)
        now = time.time()
        history = [(t, v) for t, v in self.brightness_history.get(key, []) if now - t < self.flicker_window]
        history.append((now, mean_value))
        self.brightness_history[key] = history

        if len(history) < 3:
            return 0.0

        values = np.array([v for _, v in history])
        std = float(np.std(values))
        return min(1.0, std / 25.0)

    def _is_likely_pet(self, bbox, frame_height) -> bool:
        x, y, w, h = bbox
        if w == 0:
            return False
        aspect_ratio = h / w
        relative_height = h / frame_height
        return relative_height < 0.35 and aspect_ratio < 1.2

    def _contour_features(self, contour, frame_hsv, is_ir_mode=False):
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 1
        solidity = area / hull_area
        aspect_ratio = h / w if w > 0 else 0

        blob_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = contour - [x, y]
        cv2.drawContours(blob_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

        roi = None
        roi_pixels = None
        if frame_hsv is not None:
            roi = frame_hsv[y:y + h, x:x + w]
            if roi.size > 0:
                roi_pixels = roi[blob_mask > 0]

        blob_is_grayscale = is_ir_mode
        if roi_pixels is not None and roi_pixels.size > 0 and not is_ir_mode:
            blob_is_grayscale = np.percentile(roi_pixels[:, 1], 90) < 25

        hazard_color_score = 0.0
        if not blob_is_grayscale and roi is not None and roi.size > 0:
            hue2d = roi[:, :, 0]
            sat2d = roi[:, :, 1]
            val2d = roi[:, :, 2]
            fire_mask_2d = (((hue2d < 25) | (hue2d > 168)) & (sat2d > 130) & (val2d > 180)).astype(np.uint8)
            fire_mask_2d = fire_mask_2d & (blob_mask > 0)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fire_mask_2d, connectivity=8)
            largest_cluster_area = 0
            largest_cluster_mask = None
            if num_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                largest_idx = int(np.argmax(areas)) + 1
                largest_cluster_area = int(areas[largest_idx - 1])
                largest_cluster_mask = (labels == largest_idx)

            MIN_CLUSTER_PIXELS = 40
            if largest_cluster_area >= MIN_CLUSTER_PIXELS and largest_cluster_mask is not None:
                color_score = min(1.0, largest_cluster_area / 300.0)
                mean_value = float(np.mean(val2d[largest_cluster_mask]))
                flicker = self._flicker_score(
                    (x, y, w, h), frame_hsv.shape[:2] if frame_hsv is not None else (480, 640), mean_value
                )
                hazard_color_score = color_score * (0.3 + 0.7 * flicker)
        elif blob_is_grayscale:
            hazard_color_score = max(0.0, 1.0 - solidity) * 0.5

        return {
            "bbox": (x, y, w, h),
            "area": area,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "hazard_color_score": hazard_color_score,
        }

    def score_clip(self, contours, frame_bgr, motion_area_threshold, is_ir_mode=False, light_classifier=None, person_detector=None):
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV) if not is_ir_mode else None

        model_person_score = None
        if light_classifier is not None:
            preds = light_classifier.predict(frame_bgr)
            model_person_score = preds.get("person", 0.0)

        # Get each detected person's own box (not just a single frame-wide
        # score) so we can check THEIR location against detection_zones,
        # instead of proxying through wherever a motion contour happened to be.
        person_boxes = []
        if person_detector is not None:
            person_boxes = person_detector.predict_boxes(frame_bgr)

        person_score = 0.0
        pet_detected = False
        repetitive_detected = False
        valid_blob_found = False

        for c in contours:
            area = cv2.contourArea(c)
            if area < motion_area_threshold:
                continue
            valid_blob_found = True

            feats = self._contour_features(c, frame_hsv, is_ir_mode=is_ir_mode)
            bbox = feats["bbox"]

            if self.ignore_pets and self._is_likely_pet(bbox, frame_h):
                pet_detected = True
                continue

            if not self._bbox_center_in_detection_zone(bbox):
                continue

            using_model_scores = light_classifier is not None or person_detector is not None
            if using_model_scores:
                if person_detector is not None:
                    candidate_person_score = max(
                        (b["confidence"] for b in person_boxes
                         if self._bbox_overlaps_detection_zone(b["bbox"], (frame_h, frame_w))),
                        default=0.0,
                    )
                else:
                    candidate_person_score = model_person_score if model_person_score is not None else 0.0
            else:
                aspect_ok = 1.5 <= feats["aspect_ratio"] <= 4.0
                solidity_ok = feats["solidity"] > 0.5
                candidate_person_score = 0.6 + 0.4 * feats["solidity"] if (aspect_ok and solidity_ok) else 0.0

            repetitive = self._is_repetitive(bbox, (frame_h, frame_w))
            has_real_signal = candidate_person_score > 0.15
            if repetitive and not has_real_signal:
                repetitive_detected = True
                continue

            person_score = max(person_score, candidate_person_score)

        return {
            "person_score": round(float(person_score), 3),
            "run_pipeline_b": bool(person_score > self.person_threshold),
            "is_pet_filtered": bool(pet_detected),
            "is_repetitive_filtered": bool(repetitive_detected),
            "valid_blob_found": bool(valid_blob_found),
        }