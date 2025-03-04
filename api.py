import json
import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from file_util import write_to_binary_file, create_output_folder
from summarize_full import summarize_whole
from summarize_sections import process_transcription
from transcribe import generate_transcription, split_transcription

env_path = os.path.join(".env")
load_dotenv(dotenv_path=env_path)

CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")

# fastapi dev api.py

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


@app.post("/summarize")
async def transcribe_summarize_video(
    file: UploadFile = File(...), sections: str = Form(...)
):
    video_path = f"temp_{file.filename}"
    await write_to_binary_file(video_path, file)

    output_folder = create_output_folder(file.filename)

    # Decide if whole video or sections are given
    section_titles = json.loads(sections)

    full_video_text_file = None
    if section_titles == [""]:
        transcription_file, full_video_text_file = generate_transcription(
            video_path, output_folder
        )
    else:
        transcription_file = generate_transcription(
            video_path, output_folder, section_titles
        )

    # Removes temp video file after transcription
    os.remove(video_path)

    transcription_chunks = split_transcription(transcription_file)

    summary = None

    if section_titles == [""]:
        summary = summarize_whole(full_video_text_file, output_folder)
    else:
        print(f"api.py - section titles: {section_titles}")
        summary = process_transcription(transcription_chunks, section_titles)

    return {
        "file_name": os.path.basename(transcription_file),
        "transcription": transcription_chunks,
        "summary": summary,
    }


@app.post("/post_to_confluence")
async def create_confluence_page(
    parent_id: str = Form(...),
    title: str = Form(...),
    content: str = Form(...),
    space_key: str = Form(...),
    api_token: str = Form(...),
):
    try:
        response = requests.post(
            f"{CONFLUENCE_BASE_URL}/rest/api/content",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            json={
                "type": "page",
                "title": title,
                "space": {"key": space_key},
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


@app.put("/update_on_confluence")
async def update_confluence_page(
    title: str = Form(...),
    content: str = Form(...),
    page_id: str = Form(...),
    space_key: str = Form(...),
    api_token: str = Form(...),
):
    try:
        # Get current page version to increment it
        response = requests.get(
            f"{CONFLUENCE_BASE_URL}/rest/api/content/{page_id}",
            headers={
                "Authorization": f"Bearer {api_token}",
            },
        )
        if response.ok:
            page = response.json()
            version = page["version"]["number"]
            current_title = page["title"]
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to get page: {response.text}",
            )

        # To avoid 422 use current title if new title is the same or use new if it is different.
        final_title = (
            title if title and title != "" and title != current_title else current_title
        )

        response = requests.put(
            f"{CONFLUENCE_BASE_URL}/rest/api/content/{page_id}",
            headers={
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json",
            },
            json={
                "version": {"number": version + 1},
                "title": final_title,
                "type": "page",
                "space": {"key": space_key},
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
