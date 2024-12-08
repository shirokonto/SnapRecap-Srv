import json
import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from header_detector import detect_section_headers
from summarize import process_transcription
from transcribe import process_video, split_transcription

env_path = os.path.join(".env")
load_dotenv(dotenv_path=env_path)

CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_PARENT_PAGE_ID = os.getenv("CONFLUENCE_PARENT_PAGE_ID")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")
CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_TOKEN")
CONFLUENCE_EMAIL = os.getenv("CONFLUENCE_EMAIL")

# uvicorn api:app --reload --host localhost
# uvicorn api:app --reload --host localhost --log-level debug

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
async def transcribe_video(file: UploadFile = File(...), sections: str = Form(...)):
    video_path = f"temp_{file.filename}"
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Creates output folder for files
    output_folder = os.path.join("output", file.filename)
    os.makedirs(output_folder, exist_ok=True)

    transcription_file = process_video(video_path, output_folder)

    # Reads content of the subtitle file for output in UI
    with open(transcription_file, "r") as f:
        subtitle_content = f.read()

    # Removes temp video file after transcription
    os.remove(video_path)

    section_titles = json.loads(sections)
    print(f"Section titles: {section_titles}")

    transcription_chunks = split_transcription(subtitle_content)
    detected_sections = detect_section_headers(transcription_chunks, section_titles)
    print(f"header_detector: Detected sections: {detected_sections}")

    summary = process_transcription(transcription_file)

    return {
        "file_name": os.path.basename(transcription_file),
        "transcription": subtitle_content,
        "summary": summary,
    }


@app.post("/postonconfluence")
async def create_confluence_page(
    parent_id: str = Form(...), title: str = Form(...), content: str = Form(...)
):
    try:
        response = requests.post(
            f"{CONFLUENCE_BASE_URL}/rest/api/content",
            headers={
                "Authorization": f"Bearer {CONFLUENCE_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "type": "page",
                "title": title,
                "space": {"key": CONFLUENCE_SPACE_KEY},
                "ancestors": [{"id": parent_id}],
                "body": {
                    "storage": {
                        "value": content,
                        "representation": "storage",
                    }
                },
            },
        )

        print("Response status code:", response.status_code)
        print("Response body:", response.text)

        if response.ok:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to create page: {response.text}",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while creating page: {e}")


@app.put("/updateonconfluence")
async def update_confluence_page(
    title: str = Form(...),
    content: str = Form(...),
    page_id: str = Form(...),
):
    try:
        response = requests.get(
            f"{CONFLUENCE_BASE_URL}/rest/api/content/{page_id}",
            headers={
                "Authorization": f"Bearer {CONFLUENCE_TOKEN}",
            },
        )
        if response.ok:
            page = response.json()
            version = page["version"]["number"]
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to get page: {response.text}",
            )

        response = requests.put(
            f"{CONFLUENCE_BASE_URL}/rest/api/content/{page_id}",
            headers={
                "Authorization": f"Bearer {CONFLUENCE_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "version": {
                    "number": version + 1,
                },
                "title": title,
                "type": "page",
                "space": {"key": CONFLUENCE_SPACE_KEY},
                "body": {
                    "storage": {
                        "value": content,
                        "representation": "storage",
                    }
                },
            },
        )

        if response.ok:
            return response.json()
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to update page: {response.text}",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error while updating page: {e}")
