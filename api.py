import json
import logging
import os

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from header_detector import detect_section_headers
from summarize import process_transcription, get_summary_bart, recursive_summarization
from text_chunks import read_file, split_text_into_chunks
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
    with open(video_path, "wb") as f:
        f.write(await file.read())

    # Creates output folder for files
    output_folder = os.path.join("output", file.filename)
    os.makedirs(output_folder, exist_ok=True)

    # Decide if whole video or sections are given
    section_titles = json.loads(sections)
    print(f"Section titles: {section_titles}")
    full_video_text_file = None
    if section_titles == [""]:
        transcription_file, full_video_text_file = generate_transcription(
            video_path, output_folder
        )
    else:
        transcription_file = generate_transcription(
            video_path, output_folder, section_titles
        )

    # Reads content of the transcription file for output in UI
    with open(transcription_file, "r") as f:
        subtitle_content = f.read()

    # Removes temp video file after transcription
    os.remove(video_path)

    transcription_chunks = split_transcription(subtitle_content)

    summary = None

    if section_titles == [""]:
        print(f"api: I assume a whole video since sections is empty: {section_titles}")
        long_transcript = read_file(full_video_text_file)

        if long_transcript:
            # Split text into manageable chunks
            text_chunks = split_text_into_chunks(long_transcript, max_tokens=4000)
            logging.info(f"Text chunks: {text_chunks}")

            # Generate summaries for each chunk
            summary_full = get_summary_bart(text_chunks)

            # If the summary is too long, apply another summarization pass
            if len(summary_full.split()) > 5000:
                # smaller_chunks = split_text_into_chunks(summary_full, max_tokens=1000)
                # short_summary = get_summary_bart(smaller_chunks)
                # summary = f"Section: Video Summary\nSummary: {short_summary}"
                summary_full = recursive_summarization(summary_full)
                summary = f"Section: Video Summary\nSummary: {summary_full}"
            else:
                summary = f"Section: Video Summary\nSummary: {summary_full}"

            logging.info("Summary generation complete.")
        else:
            logging.error("Error: Unable to process the text.")
        # summary = process_transcription(full_video_text_file)
    else:
        print(
            f"api: I assume video summarization should be split in sections: {section_titles}"
        )
        detected_sections = detect_section_headers(transcription_chunks, section_titles)
        print(f"header_detector: Detected sections: {detected_sections}")

        summary = process_transcription(
            transcription_file, sections=detected_sections
        )  # TODO don't use section titles

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
