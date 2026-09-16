"""Application entry point for facial-expression video analysis.

Run `python app.py`.  Configuration, processing, and output paths are kept in
video_emotion_pipeline.py; that module automatically loads GROQ_API_KEY from
the local .env file.
"""

from video_emotion_pipeline import main


if __name__ == "__main__":
    main()
