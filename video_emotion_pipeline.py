from __future__ import annotations

import csv
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import requests
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT = Path(__file__).resolve().parent


def load_local_env(path: Path) -> None:
    """Load simple KEY=value entries without overriding real environment values."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


load_local_env(ROOT / ".env")
MODEL_PATH = ROOT / "Models" / "face_landmarker.task"
FACE_DETECTOR_MODEL_PATH = ROOT / "Models" / "blaze_face_short_range.tflite"
INPUT_VIDEO = ROOT / "input.mp4"
RESULTS_DIR = ROOT / "Results"
# Layer 2: MediaPipe Face Landmarker outputs raw landmark coordinates.
LANDMARK_CSV = RESULTS_DIR / "results-layer_2.csv"
# Layer 3: MediaPipe outputs the 52 facial-blendshape feature vector per frame.
BLENDSHAPE_CSV = RESULTS_DIR / "results-layer_3.csv"
# Layer 4: Groq results joined with the corresponding windowed blendshape data.
COMBINED_CSV = RESULTS_DIR / "results-layer_4.csv"
OUTPUT_VIDEO = RESULTS_DIR / "results-layer_4.mp4"

MAX_FACES = 1
TARGET_FPS = 15.0            # Analyse and render at this rate, even if input is 30/60 FPS.
CHUNK_DURATION_MS = 3000     # Process the video in consecutive three-second chunks.
WINDOW_MS = 500              # One stable emotion conclusion twice per second.
GROQ_BATCH_DURATION_MS = CHUNK_DURATION_MS
GROQ_REQUEST_INTERVAL_SECONDS = 30
GROQ_MISSING_RESULT_RETRIES = 3
GROQ_JSON_VALIDATION_RETRIES = 3
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# MediaPipe Face Landmarker emits these 52 ARKit-compatible blendshapes.
BLENDSHAPE_COLUMNS = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft",
    "jawOpen", "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight", "mouthPressLeft",
    "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft",
    "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]
LANDMARK_COUNT = 478
LANDMARK_COLUMNS = [
    coordinate
    for point_index in range(LANDMARK_COUNT)
    for coordinate in (f"landmark_{point_index}_x", f"landmark_{point_index}_y", f"landmark_{point_index}_z")
]


def last_saved_timestamp(path: Path) -> int:
    """Return the latest timestamp in a CSV result file, or -1 when it is new."""
    if not path.is_file() or path.stat().st_size == 0:
        return -1
    latest = -1
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                latest = max(latest, int(row["timestamp_ms"]))
            except (KeyError, TypeError, ValueError):
                continue
    return latest


def extract_blendshapes(end_timestamp_ms: int | None = None) -> None:
    """Write raw landmarks and one complete 52-vector for each detected frame."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing MediaPipe model: {MODEL_PATH}")
    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Missing input video: {INPUT_VIDEO}")
    RESULTS_DIR.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {INPUT_VIDEO}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    processing_fps = min(TARGET_FPS, source_fps)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    landmark_last_timestamp = last_saved_timestamp(LANDMARK_CSV)
    blendshape_last_timestamp = last_saved_timestamp(BLENDSHAPE_CSV)
    landmark_has_data = landmark_last_timestamp >= 0
    blendshape_has_data = blendshape_last_timestamp >= 0
    if landmark_has_data or blendshape_has_data:
        print(
            "Resuming extraction after "
            f"landmarks={landmark_last_timestamp} ms, blendshapes={blendshape_last_timestamp} ms"
        )
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=MAX_FACES,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    frame_number = 0
    next_sample_ms = 0.0
    with (
        LANDMARK_CSV.open("a" if landmark_has_data else "w", newline="", encoding="utf-8") as landmark_handle,
        BLENDSHAPE_CSV.open("a" if blendshape_has_data else "w", newline="", encoding="utf-8") as blendshape_handle,
    ):
        landmark_writer = csv.DictWriter(
            landmark_handle,
            fieldnames=["frame", "timestamp_ms", "face_id"] + LANDMARK_COLUMNS,
        )
        blendshape_writer = csv.DictWriter(
            blendshape_handle,
            fieldnames=["frame", "timestamp_ms", "face_id"] + BLENDSHAPE_COLUMNS,
        )
        if not landmark_has_data:
            landmark_writer.writeheader()
        if not blendshape_has_data:
            blendshape_writer.writeheader()
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp_ms = round(frame_number * 1000 / source_fps)
                if end_timestamp_ms is not None and timestamp_ms >= end_timestamp_ms:
                    break
                # Retain only evenly spaced samples at 15 FPS (or every frame
                # when the source is already slower than 15 FPS).
                if timestamp_ms + 0.01 < next_sample_ms:
                    frame_number += 1
                    continue
                next_sample_ms += 1000 / processing_fps
                if (
                    timestamp_ms <= landmark_last_timestamp
                    and timestamp_ms <= blendshape_last_timestamp
                ):
                    frame_number += 1
                    continue
                image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(image, timestamp_ms)
                for face_id, categories in enumerate(result.face_blendshapes):
                    values = {category.category_name: float(category.score) for category in categories}
                    row = {"frame": frame_number, "timestamp_ms": timestamp_ms, "face_id": face_id}
                    row.update({name: values.get(name, 0.0) for name in BLENDSHAPE_COLUMNS})
                    if timestamp_ms > blendshape_last_timestamp:
                        blendshape_writer.writerow(row)
                    landmarks = result.face_landmarks[face_id]
                    landmark_row = {"frame": frame_number, "timestamp_ms": timestamp_ms, "face_id": face_id}
                    landmark_row.update({
                        f"landmark_{index}_{axis}": getattr(landmark, axis)
                        for index, landmark in enumerate(landmarks)
                        for axis in ("x", "y", "z")
                    })
                    if timestamp_ms > landmark_last_timestamp:
                        landmark_writer.writerow(landmark_row)
                frame_number += 1
                if frame_number % 60 == 0:
                    print(f"Extracting vectors: {frame_number}/{total}")
    capture.release()
    print(f"Saved {LANDMARK_CSV}")
    print(f"Saved {BLENDSHAPE_CSV}")


def load_temporal_windows() -> list[dict]:
    """Average every input vector in each 500ms window; no frame is discarded."""
    buckets: dict[int, list[dict]] = defaultdict(list)
    with BLENDSHAPE_CSV.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            buckets[int(row["timestamp_ms"]) // WINDOW_MS].append(row)
    windows = []
    for index in sorted(buckets):
        rows = buckets[index]
        averages = {
            name: round(sum(float(row[name]) for row in rows) / len(rows), 5)
            for name in BLENDSHAPE_COLUMNS
        }
        windows.append({"window_id": index, "timestamp_ms": index * WINDOW_MS, "vectors_used": len(rows), "blendshapes": averages})
    return windows


def request_groq_results(windows: list[dict], headers: dict[str, str]) -> dict[int, dict]:
    """Request classifications for a small set of windows and index them by ID."""
    prompt = (
        "You are classifying visible facial expression from MediaPipe ARKit blendshape values. "
        "Classify every record using one concise emotion label (for example neutral, happy, sad, angry, "
        "surprised, fearful, disgusted, contempt, or uncertain). Do not diagnose mental state. "
        "Return JSON only: {\"results\":[{\"window_id\":number,\"emotion\":string,\"confidence\":number}]}. "
        "confidence must be a number from 0 to 1. Return every supplied window_id exactly once.\n"
        + json.dumps(windows, separators=(",", ":"))
    )
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.01,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    response = None
    for attempt in range(GROQ_JSON_VALIDATION_RETRIES + 1):
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=120)
        if response.ok:
            break
        is_json_validation_failure = (
            response.status_code == 400
            and "json_validate_failed" in response.text
        )
        if not is_json_validation_failure or attempt == GROQ_JSON_VALIDATION_RETRIES:
            raise RuntimeError(f"Groq error {response.status_code}: {response.text[:500]}")
        print(
            "Groq returned an invalid structured generation; retrying "
            f"({attempt + 1}/{GROQ_JSON_VALIDATION_RETRIES})."
        )
        time.sleep(0.5)
    if response is None:  # Defensive guard for type checkers and unexpected control flow.
        raise RuntimeError("Groq request did not produce a response")
    try:
        results = json.loads(response.json()["choices"][0]["message"]["content"])["results"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Groq returned an invalid structured response") from exc
    requested_ids = {window["window_id"] for window in windows}
    return {
        int(item["window_id"]): item
        for item in results
        if int(item["window_id"]) in requested_ids
    }


def classify_with_groq() -> bool:
    """Classify unfinished windows and return whether a Groq request was made."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Set it in your shell before running this script.")
    windows = load_temporal_windows()
    if not windows:
        raise RuntimeError("No faces were detected, so there are no vectors to classify.")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    combined_fieldnames = ["timestamp_ms", "emotion", "confidence", "vectors_used"] + BLENDSHAPE_COLUMNS
    completed_window_ids = set()
    if COMBINED_CSV.is_file() and COMBINED_CSV.stat().st_size > 0:
        with COMBINED_CSV.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    completed_window_ids.add(int(row["timestamp_ms"]) // WINDOW_MS)
                except (KeyError, TypeError, ValueError):
                    continue
        print(f"Resuming classification: {len(completed_window_ids)} window(s) already saved.")
    else:
        # Create Layer 4 immediately, then checkpoint every successful Groq batch.
        with COMBINED_CSV.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=combined_fieldnames).writeheader()
    start = 0
    made_groq_request = False
    while start < len(windows):
        batch_start_ms = windows[start]["timestamp_ms"]
        end = start
        while end < len(windows) and windows[end]["timestamp_ms"] < batch_start_ms + GROQ_BATCH_DURATION_MS:
            end += 1
        batch = windows[start:end]
        unfinished_batch = [
            window for window in batch
            if window["window_id"] not in completed_window_ids
        ]
        if not unfinished_batch:
            start = end
            print(f"Classified windows: {start}/{len(windows)} (already saved)")
            continue
        made_groq_request = True
        returned: dict[int, dict] = {}
        pending = unfinished_batch
        for attempt in range(GROQ_MISSING_RESULT_RETRIES + 1):
            returned.update(request_groq_results(pending, headers))
            pending = [window for window in pending if window["window_id"] not in returned]
            if not pending:
                break
            if attempt == GROQ_MISSING_RESULT_RETRIES:
                break
            print(
                f"Groq omitted {len(pending)} window(s); retrying "
                f"({attempt + 1}/{GROQ_MISSING_RESULT_RETRIES})."
            )
            time.sleep(0.25)
        if pending:
            missing_ids = ", ".join(str(window["window_id"]) for window in pending)
            raise RuntimeError(
                f"Groq omitted window(s) after {GROQ_MISSING_RESULT_RETRIES} retries: {missing_ids}"
            )
        batch_output = []
        for window in unfinished_batch:
            item = returned.get(window["window_id"])
            batch_output.append({
                "timestamp_ms": window["timestamp_ms"],
                "emotion": str(item.get("emotion", "uncertain")).strip() or "uncertain",
                "confidence": float(item.get("confidence", 0)),
                "vectors_used": window["vectors_used"],
            })
        with COMBINED_CSV.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=combined_fieldnames)
            for result, window in zip(batch_output, unfinished_batch):
                writer.writerow({**result, **window["blendshapes"]})
        completed_window_ids.update(window["window_id"] for window in unfinished_batch)
        start = end
        print(f"Classified windows: {start}/{len(windows)}")
        time.sleep(0.1)
    print(f"Saved {COMBINED_CSV}")
    return made_groq_request


def render_emotion_video(end_timestamp_ms: int | None = None) -> None:
    """Draw a BlazeFace box and the latest timestamped emotion above it."""
    with COMBINED_CSV.open(newline="", encoding="utf-8") as handle:
        timeline = list(csv.DictReader(handle))
    if not timeline:
        raise RuntimeError("Emotion CSV is empty")
    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    output_fps = min(TARGET_FPS, source_fps)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("Could not create output video")
    if not FACE_DETECTOR_MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing BlazeFace model: {FACE_DETECTOR_MODEL_PATH}")
    detector_options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_DETECTOR_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )
    next_index, current = 0, timeline[0]
    frame = 0
    next_sample_ms = 0.0
    with vision.FaceDetector.create_from_options(detector_options) as detector:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            timestamp_ms = round(frame * 1000 / source_fps)
            if end_timestamp_ms is not None and timestamp_ms >= end_timestamp_ms:
                break
            if timestamp_ms + 0.01 < next_sample_ms:
                frame += 1
                continue
            next_sample_ms += 1000 / output_fps
            while next_index + 1 < len(timeline) and int(timeline[next_index + 1]["timestamp_ms"]) <= timestamp_ms:
                next_index += 1
                current = timeline[next_index]

            rgb_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            )
            detections = detector.detect_for_video(rgb_image, timestamp_ms).detections
            if detections:
                detection = max(detections, key=lambda item: item.categories[0].score)
                box = detection.bounding_box
                left = max(0, box.origin_x)
                top = max(0, box.origin_y)
                right = min(width - 1, box.origin_x + box.width)
                bottom = min(height - 1, box.origin_y + box.height)
                cv2.rectangle(image, (left, top), (right, bottom), (80, 255, 80), 2)
                label = f"{current['emotion']} ({float(current['confidence']):.0%})"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                label_baseline_y = max(label_height + baseline + 8, top - 8)
                label_right = min(width - 1, left + label_width + 12)
                cv2.rectangle(
                    image,
                    (left, label_baseline_y - label_height - baseline - 8),
                    (label_right, label_baseline_y + baseline),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    image,
                    label,
                    (left + 6, label_baseline_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (80, 255, 80),
                    2,
                    cv2.LINE_AA,
                )
            writer.write(image)
            frame += 1
    capture.release()
    writer.release()
    print(f"Saved {OUTPUT_VIDEO}")


def input_duration_ms() -> int:
    """Return the duration of the configured input video in milliseconds."""
    if not INPUT_VIDEO.is_file():
        raise FileNotFoundError(f"Missing input video: {INPUT_VIDEO}")
    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {INPUT_VIDEO}")
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.release()
    return round(frame_count * 1000 / fps)


def main() -> None:
    duration_ms = input_duration_ms()
    for chunk_end_ms in range(CHUNK_DURATION_MS, duration_ms + CHUNK_DURATION_MS, CHUNK_DURATION_MS):
        current_end_ms = min(chunk_end_ms, duration_ms)
        print(f"Processing video through {current_end_ms} ms of {duration_ms} ms")
        extract_blendshapes(end_timestamp_ms=chunk_end_ms)
        made_groq_request = classify_with_groq()
        # An MP4 cannot safely be appended in place, so rebuild the preview from
        # the checkpointed Layer 4 CSV through the newest completed chunk.
        render_emotion_video(end_timestamp_ms=current_end_ms)
        if made_groq_request and current_end_ms < duration_ms:
            print(f"Waiting {GROQ_REQUEST_INTERVAL_SECONDS}s before the next Groq request.")
            time.sleep(GROQ_REQUEST_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
