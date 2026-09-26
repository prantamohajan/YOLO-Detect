# Detection Lab

A web frontend for real-time YOLO object detection, built on top of the original CLI script. It streams the webcam through a live-annotated feed in the browser, with an adjustable confidence threshold, live FPS/object-count readout, snapshot capture, and video recording — no OpenCV desktop window required.

## Project structure

```
yolo-detect-web/
├── app.py
├── templates/
│   └── index.html
├── outputs/          (created automatically for snapshots/recordings)
└── README.md
```

## Requirements

```bash
pip install flask opencv-python ultralytics
```

A webcam connected to the machine running `app.py`.

## Running it

```bash
python app.py
```

Then open `http://localhost:5001`. The default model is `yolov8s.pt` (downloaded automatically by Ultralytics on first run) and the default camera source is index `0`.

To use a different model or camera source without editing code:

```bash
YOLO_MODEL=yolov8n.pt YOLO_SOURCE=1 python app.py
```

## Using the dashboard

- **Confidence threshold** — drag the slider to change the minimum detection confidence live, without restarting the stream.
- **Session panel** — shows the active model and which classes are currently visible in frame.
- **Capture** — "Save snapshot" writes the current annotated frame to `outputs/`; "Start/Stop recording" writes an `.mp4` of the annotated stream to the same folder.

## Detecting an object the model doesn't know

`yolov8s.pt` only recognizes the 80 COCO classes. For an uncommon object outside that set:

1. Collect and label a dataset for that object (tools like Roboflow or CVAT work well for this).
2. Fine-tune a YOLOv8 model on it with Ultralytics: `yolo train data=your_dataset.yaml model=yolov8s.pt epochs=100`.
3. Point the app at the resulting checkpoint: `YOLO_MODEL=runs/detect/train/weights/best.pt python app.py`.

No other changes to `app.py` or the frontend are needed — the model path is the only thing that changes.

## Notes on changes from the original CLI script

- Replaced `cv2.imshow`/`cv2.waitKey` (desktop-only) with an MJPEG stream served over Flask, so it works in a browser and can run headless.
- The confidence threshold is now adjustable at runtime from the UI instead of being fixed at startup via `--conf`.
- FPS and object count are exposed through a `/stats` endpoint and rendered as HTML instead of being burned into the video frame with `cv2.putText`.
- Snapshot and recording are now triggered by UI buttons (`/snapshot`, `/record/start`, `/record/stop`) instead of keyboard shortcuts on the OpenCV window.
- The camera is opened lazily and reopened after a run of failed reads, instead of only being grabbed once at startup.

## Limitations

- One shared camera and detection state — fine for a single-user demo, not for multiple people running independent sessions at once.
- Inference speed depends on your hardware; a CPU-only machine will see meaningfully lower FPS than a GPU.