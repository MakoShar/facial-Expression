# Facial Expression Video Analysis

This project analyses a video of one face and produces an annotated copy showing the inferred visible facial expression over time. It combines local face-landmark processing with a Groq model for short-window emotion labels. Each pipeline layer saves its result in `Results/`.

## How it works

1. MediaPipe Face Landmarker reads `input.mp4` and extracts its 52 ARKit-compatible facial blendshape scores.
2. Samples are taken at up to 15 FPS and averaged into 500 ms temporal windows.
3. Each five-second batch of windows is sent to Groq for a JSON-only expression label and confidence score.
4. The inferred label is drawn onto a 15 FPS output video.

The classifier is intended to describe **visible facial expression** (such as `happy`, `sad`, `angry`, or `neutral`) and is not a mental-state or clinical diagnostic tool.

## Project layout

```text
app.py                         # Simple application entry point
video_emotion_pipeline.py      # Extraction, classification, and rendering pipeline
input.mp4                      # Video to analyse (replace with your own)
Models/face_landmarker.task    # Required MediaPipe Face Landmarker model
Models/blaze_face_short_range.tflite # BlazeFace model for boxes in the rendered video
.env                           # Local Groq credentials; intentionally not committed
Results/                       # Generated layer-by-layer CSVs and video
```

## Requirements

- Python 3.10 or later
- A Groq API key
- The MediaPipe task model at `Models/face_landmarker.task`

Install the Python packages:

```bash
pip install opencv-python mediapipe requests
```

## Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
# Optional. Defaults to openai/gpt-oss-20b.
# GROQ_MODEL=your_supported_groq_model
```

`GROQ_API_KEY` may also be set in the shell environment. Environment values take precedence over entries in `.env`.

## Run

Place the video to analyse at `input.mp4`, then run:

```bash
python app.py
```

The pipeline runs all three stages in order: blendshape extraction, Groq classification, and video rendering. It processes a maximum of one face per frame.

## Output

Generated files are written under `Results/`:

- `results-layer_2.csv` - Layer 2 output: raw MediaPipe facial-landmark coordinates, with frame and timestamp.
- `results-layer_3.csv` - Layer 3 output: timestamped 52-feature facial-blendshape vectors.
- `results-layer_4.csv` - Layer 4 output: Groq emotion results combined with the corresponding 500 ms blendshape window values.
- `results-layer_4.mp4` - annotated video with a BlazeFace bounding box and emotion label above the detected face.

The CSV outputs are resumable. On a later run, the pipeline keeps the existing rows, appends only new Layer 2 and Layer 3 timestamps, and sends only Layer 4 time windows that do not already have a saved Groq result. After each completed chunk, the MP4 preview is regenerated through that chunk from the accumulated Layer 4 CSV before the Groq cooldown begins.

## Pipeline settings

The main settings are at the top of `video_emotion_pipeline.py`:

- `TARGET_FPS = 15.0`
- `WINDOW_MS = 500`
- `GROQ_BATCH_DURATION_MS = 5000`
- `MAX_FACES = 1`
- `CHUNK_DURATION_MS = 3000` (processes consecutive three-second chunks)
- `GROQ_REQUEST_INTERVAL_SECONDS = 30` (waits 30 seconds between Groq requests)

Adjust these if you need a different speed/detail trade-off. Shorter windows and higher sampling rates provide more detail but increase data volume and API use.

## Notes

- The input must be a video format OpenCV can read; MP4 is the included workflow.
- A frame with no detected face does not add a blendshape row. If no faces are detected at all, classification stops with an error.
- Never commit `.env` or API keys. The provided `.gitignore` excludes it.
