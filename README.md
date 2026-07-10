# Real-Time Object Detection

Real-time object detection system built with YOLOv8 and OpenCV. Detects 80 object classes (vehicles, people, animals, and more) from a live webcam feed.

## Features

- Real-time detection via webcam
- Live FPS counter
- Adjustable confidence threshold
- Screenshot capture (press `s`)
- Video recording mode (`--save`)

## Setup

```bash
conda create -n objdetect python=3.10 -y
conda activate objdetect
pip install -r requirements.txt
```

## Usage

```bash
python detect.py
```

### Optional arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | YOLOv8 model variant (n/s/m/l/x) | `yolov8s.pt` |
| `--source` | Camera index | `0` |
| `--conf` | Confidence threshold | `0.4` |
| `--save` | Save output as video | `False` |

Example:

```bash
python detect.py --model yolov8m.pt --conf 0.3 --save
```

## Controls

- `q` — quit
- `s` — save a snapshot

## Tech Stack

- YOLOv8 (Ultralytics)
- OpenCV
- Python

## Roadmap

- [ ] Object tracking with ByteTrack
- [ ] Vehicle/object counting
- [ ] Custom fine-tuned classes
- [ ] Web app version (FastAPI + React)
