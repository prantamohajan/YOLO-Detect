import cv2
import time
import argparse
import os
from datetime import datetime
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection")
    parser.add_argument("--model", type=str, default="yolov8s.pt")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print("Webcam dosen't work or cannot be opened.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        os.makedirs("outputs", exist_ok=True)
        filename = f"outputs/record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=args.conf, verbose=False)
        annotated_frame = results[0].plot()

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        detected_count = len(results[0].boxes)

        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Objects: {detected_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Real-Time Object Detection", annotated_frame)

        if writer is not None:
            writer.write(annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            os.makedirs("outputs", exist_ok=True)
            snap_name = f"outputs/snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(snap_name, annotated_frame)
            print(f"Snapshot saved: {snap_name}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()