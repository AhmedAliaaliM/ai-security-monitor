import cv2
import time
from core.hazard_pipeline import HazardPipeline

pipeline = HazardPipeline()

cap = cv2.VideoCapture(0)

print("Hold the lighter in frame now. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        detections = pipeline.process_frame(frame)

        if detections:
            print(detections)

        time.sleep(0.5)

except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()