import cv2
import csv
import os
import threading
import time
import numpy as np
from datetime import datetime

from flask import Flask, render_template, Response, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename
import uuid

import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarksConnections
)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "Models/face_landmarker.task"

LIVE_FOLDER = "live"

CAMERA_INDEX = 0

MAX_FACES = 10


# ============================================================
# CREATE LIVE FOLDER
# ============================================================

os.makedirs(LIVE_FOLDER, exist_ok=True)


# ============================================================
# COLORS - OpenCV uses BGR
# ============================================================

WHITE = (220, 220, 220)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
BLACK = (0, 0, 0)


# ============================================================
# MEDIAPIPE CONNECTIONS
# ============================================================

TESSELLATION = (
    FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION
)

CONTOURS = (
    FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS
)

LEFT_EYE = (
    FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE
)

RIGHT_EYE = (
    FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE
)

LEFT_EYEBROW = (
    FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW
)

RIGHT_EYEBROW = (
    FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW
)

LIPS = (
    FaceLandmarksConnections.FACE_LANDMARKS_LIPS
)


# ============================================================
# GLOBAL STATE
# ============================================================

camera = None

landmarker = None

processing_thread = None

running = False

latest_frame = None

state_lock = threading.Lock()


# CSV state

csv_file = None
csv_writer = None

current_start_time = None
current_csv_temp = None

frame_number = 0

# Upload jobs are intentionally separate from the live-camera state.  This lets
# a recorded file be processed without interrupting an active browser session.
jobs = {}
jobs_lock = threading.Lock()
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


# ============================================================
# DRAW CONNECTIONS
# ============================================================

def draw_connections(
    frame,
    landmarks,
    connections,
    color,
    thickness=1
):

    height, width = frame.shape[:2]

    for connection in connections:

        start_idx = connection.start
        end_idx = connection.end

        if (
            start_idx >= len(landmarks)
            or
            end_idx >= len(landmarks)
        ):
            continue

        start = landmarks[start_idx]
        end = landmarks[end_idx]

        x1 = int(start.x * width)
        y1 = int(start.y * height)

        x2 = int(end.x * width)
        y2 = int(end.y * height)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA
        )


# ============================================================
# DRAW FACE WIREFRAME
# ============================================================

def draw_face_mesh(frame, landmarks):

    # Full triangular mesh
    draw_connections(
        frame,
        landmarks,
        TESSELLATION,
        WHITE,
        1
    )

    # Face outline
    draw_connections(
        frame,
        landmarks,
        CONTOURS,
        WHITE,
        1
    )

    # Left eye + eyebrow
    draw_connections(
        frame,
        landmarks,
        LEFT_EYE,
        GREEN,
        2
    )

    draw_connections(
        frame,
        landmarks,
        LEFT_EYEBROW,
        GREEN,
        2
    )

    # Right eye + eyebrow
    draw_connections(
        frame,
        landmarks,
        RIGHT_EYE,
        RED,
        2
    )

    draw_connections(
        frame,
        landmarks,
        RIGHT_EYEBROW,
        RED,
        2
    )

    # Lips
    draw_connections(
        frame,
        landmarks,
        LIPS,
        WHITE,
        2
    )


# ============================================================
# BLENDSHAPES -> DICTIONARY
# ============================================================

def blendshapes_to_dict(blendshapes):

    result = {}

    for category in blendshapes:

        result[category.category_name] = float(
            category.score
        )

    return result


# ============================================================
# GET BLENDSHAPE
# ============================================================

def get_score(blendshapes, name):

    return blendshapes.get(name, 0.0)


# ============================================================
# CALCULATE SIMPLE EXPRESSIONS
# ============================================================

def calculate_expression_values(bs):

    smile = (
        get_score(bs, "mouthSmileLeft")
        +
        get_score(bs, "mouthSmileRight")
    ) / 2

    blink = (
        get_score(bs, "eyeBlinkLeft")
        +
        get_score(bs, "eyeBlinkRight")
    ) / 2

    jaw = get_score(
        bs,
        "jawOpen"
    )

    brow = (
        get_score(bs, "browInnerUp")
        +
        get_score(bs, "browOuterUpLeft")
        +
        get_score(bs, "browOuterUpRight")
    ) / 3

    return smile, blink, jaw, brow


# ============================================================
# DRAW INFORMATION
# ============================================================

def draw_expression_info(
    frame,
    face_id,
    landmarks,
    blendshapes
):

    height, width = frame.shape[:2]

    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x = int(min(xs) * width)
    y = int(min(ys) * height)

    x = max(10, x)
    y = max(120, y)

    smile, blink, jaw, brow = (
        calculate_expression_values(
            blendshapes
        )
    )

    lines = [
        f"Face {face_id}",
        f"Smile : {smile:.2f}",
        f"Blink : {blink:.2f}",
        f"Jaw   : {jaw:.2f}",
        f"Brow  : {brow:.2f}"
    ]

    box_width = 155
    box_height = 110

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x, y - box_height),
        (x + box_width, y),
        BLACK,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame
    )

    for i, text in enumerate(lines):

        cv2.putText(
            frame,
            text,
            (
                x + 8,
                y - box_height + 22 + i * 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            WHITE,
            1,
            cv2.LINE_AA
        )


# ============================================================
# START CSV
# ============================================================

def start_csv():

    global csv_file
    global csv_writer
    global current_start_time
    global current_csv_temp
    global frame_number

    current_start_time = datetime.now()

    start_string = (
        current_start_time.strftime(
            "%Y%m%d-%H%M%S"
        )
    )

    # Temporary file while recording
    current_csv_temp = os.path.join(
        LIVE_FOLDER,
        f"{start_string}-LIVE.csv"
    )

    csv_file = open(
        current_csv_temp,
        "w",
        newline="",
        encoding="utf-8"
    )

    csv_writer = None

    frame_number = 0

    print(
        f"CSV started: {current_csv_temp}"
    )


# ============================================================
# FINISH CSV
# ============================================================

def finish_csv():

    global csv_file
    global csv_writer
    global current_start_time
    global current_csv_temp

    if csv_file is None:
        return

    csv_file.close()

    end_time = datetime.now()

    start_string = (
        current_start_time.strftime(
            "%Y%m%d-%H%M%S"
        )
    )

    end_string = (
        end_time.strftime(
            "%Y%m%d-%H%M%S"
        )
    )

    final_filename = (
        f"{start_string}-{end_string}.csv"
    )

    final_path = os.path.join(
        LIVE_FOLDER,
        final_filename
    )

    os.rename(
        current_csv_temp,
        final_path
    )

    print(
        f"CSV saved: {final_path}"
    )

    csv_file = None
    csv_writer = None
    current_start_time = None
    current_csv_temp = None


# ============================================================
# CAMERA PROCESSING
# ============================================================

def process_camera():

    global camera
    global landmarker
    global running
    global latest_frame
    global csv_writer
    global frame_number

    print("Opening webcam...")

    camera = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not camera.isOpened():

        print("ERROR: Could not open webcam.")

        running = False

        return

    # --------------------------------------------------------
    # Camera settings
    # --------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )


    # --------------------------------------------------------
    # MediaPipe
    # --------------------------------------------------------

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceLandmarkerOptions(

        base_options=base_options,

        running_mode=vision.RunningMode.VIDEO,

        num_faces=MAX_FACES,

        min_face_detection_confidence=0.5,

        min_face_presence_confidence=0.5,

        min_tracking_confidence=0.5,

        output_face_blendshapes=True,

        output_facial_transformation_matrixes=True
    )

    landmarker = (
        vision.FaceLandmarker.create_from_options(
            options
        )
    )


    # --------------------------------------------------------
    # Start CSV
    # --------------------------------------------------------

    start_csv()


    # --------------------------------------------------------
    # Processing loop
    # --------------------------------------------------------

    timestamp_ms = 0

    while running:

        success, frame = camera.read()

        if not success:

            print(
                "Could not read webcam frame."
            )

            break


        # ----------------------------------------------------
        # Flip horizontally for natural webcam view
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        # ----------------------------------------------------
        # Convert BGR -> RGB
        # ----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp_ms += 33


        # ----------------------------------------------------
        # MediaPipe
        # ----------------------------------------------------

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        detected_faces = len(
            result.face_landmarks
        )


        # ====================================================
        # PROCESS EACH FACE
        # ====================================================

        for face_id in range(detected_faces):

            landmarks = result.face_landmarks[
                face_id
            ]


            # -----------------------------------------------
            # Draw 478 landmark wireframe
            # -----------------------------------------------

            draw_face_mesh(
                frame,
                landmarks
            )


            # -----------------------------------------------
            # Blendshapes
            # -----------------------------------------------

            if (
                result.face_blendshapes
                and
                face_id < len(
                    result.face_blendshapes
                )
            ):

                blendshapes = (
                    blendshapes_to_dict(
                        result.face_blendshapes[
                            face_id
                        ]
                    )
                )


                # ===========================================
                # CREATE CSV HEADER
                # ===========================================

                if csv_writer is None:

                    blendshape_names = list(
                        blendshapes.keys()
                    )

                    csv_writer = csv.DictWriter(
                        csv_file,
                        fieldnames=[
                            "frame",
                            "timestamp_ms",
                            "face_id"
                        ]
                        +
                        blendshape_names
                    )

                    csv_writer.writeheader()


                # ===========================================
                # CSV ROW
                # ===========================================

                row = {
                    "frame": frame_number,
                    "timestamp_ms": timestamp_ms,
                    "face_id": face_id
                }


                for name in blendshape_names:

                    row[name] = blendshapes.get(
                        name,
                        0.0
                    )


                csv_writer.writerow(row)

                # Flush immediately so the CSV is truly live
                csv_file.flush()


                # ===========================================
                # Draw information
                # ===========================================

                draw_expression_info(
                    frame,
                    face_id,
                    landmarks,
                    blendshapes
                )


        # ====================================================
        # GLOBAL INFO
        # ====================================================

        cv2.rectangle(
            frame,
            (10, 10),
            (220, 75),
            BLACK,
            -1
        )

        cv2.putText(
            frame,
            "LIVE",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            RED,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Faces: {detected_faces}",
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            WHITE,
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # JPEG ENCODE
        # ====================================================

        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                85
            ]
        )

        if success:

            with state_lock:

                latest_frame = (
                    encoded.tobytes()
                )


        frame_number += 1


    # ========================================================
    # CLEANUP
    # ========================================================

    print("Stopping webcam...")

    if camera is not None:

        camera.release()

        camera = None


    if landmarker is not None:

        landmarker.close()

        landmarker = None


    finish_csv()

    print("Webcam stopped.")


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_frames():

    while True:

        with state_lock:
            frame = latest_frame

        if frame is None:

            placeholder_frame = np.zeros(
                (480, 640, 3),
                dtype=np.uint8
            )

            cv2.putText(
                placeholder_frame,
                "Waiting for webcam...",
                (150, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                WHITE,
                2,
                cv2.LINE_AA
            )

            success, encoded = cv2.imencode(
                ".jpg",
                placeholder_frame
            )

            if not success:
                continue

            frame = encoded.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )
# ============================================================
# UPLOADED VIDEO PROCESSING
# ============================================================

def process_uploaded_video(job_id, input_path, output_name, csv_name):
    """Annotate an uploaded video and retain its blendshapes in live/."""
    try:
        with jobs_lock:
            jobs[job_id].update(status="processing", progress=0)

        capture = cv2.VideoCapture(input_path)
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open this video")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        output_path = os.path.join(LIVE_FOLDER, output_name)
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create the output video")

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO, num_faces=MAX_FACES,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5, output_face_blendshapes=True
        )
        frame_index, csv_writer_local = 0, None
        csv_path = os.path.join(LIVE_FOLDER, csv_name)
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_handle:
            with vision.FaceLandmarker.create_from_options(options) as video_landmarker:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    timestamp_ms = round(frame_index * 1000 / fps)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = video_landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                    for face_id, landmarks in enumerate(result.face_landmarks):
                        draw_face_mesh(frame, landmarks)
                        if face_id < len(result.face_blendshapes):
                            blendshapes = blendshapes_to_dict(result.face_blendshapes[face_id])
                            if csv_writer_local is None:
                                names = list(blendshapes.keys())
                                csv_writer_local = csv.DictWriter(csv_handle, fieldnames=["frame", "timestamp_ms", "face_id"] + names)
                                csv_writer_local.writeheader()
                            row = {"frame": frame_index, "timestamp_ms": timestamp_ms, "face_id": face_id}
                            row.update({name: blendshapes.get(name, 0.0) for name in names})
                            csv_writer_local.writerow(row)
                            draw_expression_info(frame, face_id, landmarks, blendshapes)
                    writer.write(frame)
                    frame_index += 1
                    if frame_index % 10 == 0:
                        with jobs_lock:
                            jobs[job_id]["progress"] = round(100 * frame_index / total) if total else 0
        capture.release()
        writer.release()
        with jobs_lock:
            jobs[job_id].update(status="complete", progress=100, output=output_name, csv=csv_name)
    except Exception as exc:
        with jobs_lock:
            jobs[job_id].update(status="error", error=str(exc))


@app.route("/upload", methods=["POST"])
def upload_video():
    uploaded = request.files.get("video")
    if uploaded is None or not uploaded.filename:
        return jsonify(error="Choose a video file first."), 400
    filename = secure_filename(uploaded.filename)
    extension = os.path.splitext(filename)[1].lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        return jsonify(error="Supported files: MP4, AVI, MOV, MKV, WEBM."), 400
    job_id = uuid.uuid4().hex[:12]
    input_name = f"{job_id}-input{extension}"
    output_name = f"{job_id}-output.mp4"
    csv_name = f"{job_id}-blendshapes.csv"
    input_path = os.path.join(LIVE_FOLDER, input_name)
    uploaded.save(input_path)
    with jobs_lock:
        jobs[job_id] = {"status": "queued", "progress": 0}
    threading.Thread(target=process_uploaded_video, args=(job_id, input_path, output_name, csv_name), daemon=True).start()
    return jsonify(job_id=job_id), 202


@app.route("/jobs/<job_id>")
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return jsonify(error="Unknown job"), 404
    return jsonify(job)


@app.route("/live/<path:filename>")
def live_file(filename):
    return send_from_directory(LIVE_FOLDER, filename, as_attachment=False)


def generate_processed_frames(filename):
    """Browser-safe preview for files whose OpenCV output codec is unsupported."""
    capture = cv2.VideoCapture(os.path.join(LIVE_FOLDER, filename))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    delay = max(0.001, 1 / fps)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n")
            time.sleep(delay)
    finally:
        capture.release()


@app.route("/processed_feed/<path:filename>")
def processed_feed(filename):
    if not filename.endswith("-output.mp4") or not os.path.isfile(os.path.join(LIVE_FOLDER, filename)):
        return jsonify(error="Output video not found"), 404
    return Response(generate_processed_frames(filename), mimetype="multipart/x-mixed-replace; boundary=frame")


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )
# ============================================================
# VIDEO FEED ROUTE
# ============================================================

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ============================================================
# START
# ============================================================

@app.route(
    "/start",
    methods=["POST"]
)
def start():

    global processing_thread
    global running

    if running:

        return jsonify({
            "status": "already_running"
        })


    running = True

    processing_thread = threading.Thread(
        target=process_camera,
        daemon=True
    )

    processing_thread.start()


    return jsonify({
        "status": "started"
    })


# ============================================================
# STOP
# ============================================================

@app.route(
    "/stop",
    methods=["POST"]
)
def stop():

    global running

    if not running:

        return jsonify({
            "status": "not_running"
        })


    running = False


    return jsonify({
        "status": "stopping"
    })


# ============================================================
# STATUS
# ============================================================

@app.route("/status")
def status():

    return jsonify({
        "running": running,
        "frame": frame_number
    })


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("==============================================")
    print("       FACIAL EXPRESSION LIVE SERVER")
    print("==============================================")
    print()
    print("Open:")
    print("http://127.0.0.1:5000")
    print()
    print("==============================================")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True
    )
