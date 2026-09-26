import os
import threading
import time
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8s.pt")
SOURCE = int(os.environ.get("YOLO_SOURCE", 0))
OUTPUT_DIR = "outputs"

app = Flask(__name__)

state_lock = threading.Lock()
state = {
    "conf": 0.4,
    "fps": 0.0,
    "object_count": 0,
    "classes_present": [],
    "recording": False,
}

model = YOLO(MODEL_PATH)
camera = None
video_writer = None


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(SOURCE)
    return camera


def camera_release():
    global camera
    if camera is not None:
        camera.release()
        camera = None


def start_recording(frame_width, frame_height):
    global video_writer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(
        OUTPUT_DIR, f"record_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))
    return filename


def stop_recording():
    global video_writer
    if video_writer is not None:
        video_writer.release()
        video_writer = None


def generate_frames():
    prev_time = 0
    consecutive_failures = 0

    while True:
        cam = get_camera()
        success, frame = cam.read()

        if not success:
            consecutive_failures += 1
            if consecutive_failures > 30:
                camera_release()
                time.sleep(0.5)
                consecutive_failures = 0
            continue

        consecutive_failures = 0

        with state_lock:
            conf = state["conf"]

        results = model(frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot()

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        boxes = results[0].boxes
        detected_count = len(boxes)
        class_names = sorted(
            {model.names[int(c)] for c in boxes.cls} if detected_count else set()
        )

        with state_lock:
            state["fps"] = round(fps, 1)
            state["object_count"] = detected_count
            state["classes_present"] = class_names
            recording = state["recording"]

        if recording and video_writer is not None:
            video_writer.write(annotated_frame)

        ok, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return render_template("index.html", model_name=MODEL_PATH)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stats")
def stats():
    with state_lock:
        return jsonify(
            fps=state["fps"],
            object_count=state["object_count"],
            classes_present=state["classes_present"],
            conf=state["conf"],
            recording=state["recording"],
            model=MODEL_PATH,
        )


@app.route("/set_conf", methods=["POST"])
def set_conf():
    value = request.get_json(silent=True) or {}
    conf = value.get("conf")
    if conf is None:
        return jsonify(status="error", message="conf required"), 400
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        return jsonify(status="error", message="conf must be a number"), 400
    conf = max(0.05, min(conf, 0.95))
    with state_lock:
        state["conf"] = conf
    return jsonify(status="success", conf=conf)


@app.route("/snapshot", methods=["POST"])
def snapshot():
    cam = get_camera()
    success, frame = cam.read()
    if not success:
        return jsonify(status="error", message="camera unavailable"), 503
    with state_lock:
        conf = state["conf"]
    results = model(frame, conf=conf, verbose=False)
    annotated_frame = results[0].plot()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filename = os.path.join(
        OUTPUT_DIR, f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    )
    cv2.imwrite(filename, annotated_frame)
    return jsonify(status="success", file=filename)


@app.route("/record/start", methods=["POST"])
def record_start():
    cam = get_camera()
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    with state_lock:
        if state["recording"]:
            return jsonify(status="error", message="already recording"), 400
        filename = start_recording(frame_width, frame_height)
        state["recording"] = True
    return jsonify(status="success", file=filename)


@app.route("/record/stop", methods=["POST"])
def record_stop():
    with state_lock:
        if not state["recording"]:
            return jsonify(status="error", message="not recording"), 400
        state["recording"] = False
    stop_recording()
    return jsonify(status="success")


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
    finally:
        stop_recording()
        camera_release()