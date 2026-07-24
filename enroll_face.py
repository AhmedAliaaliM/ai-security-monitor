"""
Enrollment tool - add authorized faces to the known-faces database.

Usage:
    python enroll_face.py --name "John" --image path\to\photo.jpg
    python enroll_face.py --name "John" --webcam
"""

import argparse
import cv2

from core.identity_pipeline import IdentityPipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True, help="Name to enroll this face under")
    parser.add_argument("--image", type=str, default=None, help="Path to a clear, front-facing photo")
    parser.add_argument("--webcam", action="store_true", help="Capture a frame from webcam instead of a file")
    parser.add_argument("--db", type=str, default="known_faces.json", help="Path to the face database file")
    args = parser.parse_args()

    if not args.image and not args.webcam:
        print("Provide either --image <path> or --webcam")
        return

    pipeline = IdentityPipeline(db_path=args.db)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"Could not read image at {args.image}")
            return
    else:
        cap = cv2.VideoCapture(0)
        print("Look at the camera. Press SPACE to capture, 'q' to cancel.")
        frame = None
        while True:
            ret, preview = cap.read()
            if not ret:
                print("Could not read from webcam.")
                cap.release()
                return
            cv2.imshow("Enrollment - press SPACE to capture", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                frame = preview
                break
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("Cancelled.")
                return
        cap.release()
        cv2.destroyAllWindows()

    success = pipeline.enroll_from_image(args.name, frame)
    if success:
        print(f"Enrolled '{args.name}' successfully. Database: {args.db}")
    else:
        print("No face detected in the provided image - try a clearer, front-facing photo.")


if __name__ == "__main__":
    main()