import os

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from summarize import summarize_srt
from transcribe import process_video

#  uvicorn api:app --reload --host localhost

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):

    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Creates output folder for files
    output_folder = os.path.join("output", file.filename)
    os.makedirs(output_folder, exist_ok=True)

    # Calls the process_video function from transcribe.py
    transcription_file = process_video(video_path, output_folder)

    # Reads content of the subtitle file
    with open(transcription_file, "r") as f:
        subtitle_content = f.read()

    # Removes temp video file after transcription
    os.remove(video_path)

    summary = summarize_srt(transcription_file)

    return {
        "file_name": os.path.basename(transcription_file),
        "transcription": subtitle_content,
        "summary": summary,
    }
