"""
Person detection using BASE YOLOv8n (COCO-pretrained, class 0 = person).
Replaces the custom light classifier's person output, which was
miscalibrated on real webcam footage.
"""

PERSON_CLASS_ID = 0


class PersonDetector:
    def __init__(self, weights_path: str = "yolov8n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(weights_path)  # auto-downloads base COCO weights

    def predict(self, frame_bgr) -> float:
        """Kept for backwards compatibility - max person confidence, whole frame."""
        boxes = self.predict_boxes(frame_bgr)
        return max((b["confidence"] for b in boxes), default=0.0)

    def predict_boxes(self, frame_bgr):
        """
        Returns [{'confidence': float, 'bbox': (x, y, w, h)}, ...] in pixel
        coords, one entry per detected person. Use this (not predict()) when
        you need to know WHERE a person is, e.g. to check against
        detection_zones - predict() alone throws that location away.
        """
        results = self.model(frame_bgr, verbose=False, classes=[PERSON_CLASS_ID])[0]
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({
                "confidence": float(box.conf[0]),
                "bbox": (x1, y1, x2 - x1, y2 - y1),
            })
        return boxes