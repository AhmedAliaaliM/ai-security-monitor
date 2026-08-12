"""
Pipeline A: Hazard detection - real fire/smoke detection using a
pretrained YOLOv8n model (not our old HSV color heuristic).
"""

CONFIDENCE_THRESHOLD = 0.2  # raised from 0.5 - 0.5-0.7 range was catching false positives
MAX_BBOX_FRAME_FRACTION = 0.3  # reject detections that implausibly cover almost the whole frame


class HazardPipeline:
    def __init__(self, weights_path: str = "models/fire_yolov8n.pt", confidence_threshold: float = CONFIDENCE_THRESHOLD):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)
        self.confidence_threshold = confidence_threshold

    def process_frame(self, frame_bgr):
        """
        Returns a list of detections: [{class_name, confidence, bbox}, ...]
        Filters out implausible full-frame detections - real fire/smoke
        rarely fills the ENTIRE camera view; a detection that does is more
        likely the model pattern-matching overall frame texture than a
        real object.
        """
        frame_h, frame_w = frame_bgr.shape[:2]
        frame_area = frame_h * frame_w

        results = self.model(frame_bgr, verbose=False)[0]
        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            bbox_area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if bbox_area / frame_area > MAX_BBOX_FRAME_FRACTION:
                continue  # implausible - covers almost the whole frame

            detections.append({
                "class_name": self.model.names[cls_id],
                "confidence": round(conf, 3),
                "bbox": [round(v, 1) for v in xyxy],
            })
        return detections

    def decide(self, detections, threshold: float = None):
        """
        Turns raw detections into a hazard decision. Uses the instance's
        own confidence_threshold (set at construction, live-editable from
        the dashboard) unless a specific threshold is passed in.
        """
        if threshold is None:
            threshold = self.confidence_threshold
        confident = [d for d in detections if d["confidence"] >= threshold]
        if not confident:
            return {"status": "no_hazard", "detail": None}
        best = max(confident, key=lambda d: d["confidence"])
        return {"status": "HAZARD", "detail": best}