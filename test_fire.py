import cv2
from core.hazard_pipeline import HazardPipeline

pipeline = HazardPipeline()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = pipeline.process_frame(frame)
    print(pipeline.decide(detections))

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()