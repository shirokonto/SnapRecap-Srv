from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import os
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

    # Call the transcription function from main.py
    transcription_file = process_video(video_path)

    # Read the content of the subtitle file
    with open(transcription_file, "r") as f:
        subtitle_content = f.read()

    # Optionally remove the temp video file after transcription
    os.remove(video_path)

    return {"file_name": transcription_file, "transcription": subtitle_content}
