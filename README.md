# SnapRecap-Srv

## Overview

SnapRecap-Srv is a project designed to transcribe and summarize video content and works in combination with the SnapRecap-UI project.\
It uses `faster-whisper` for transcription and `bart-large-cnn` for summarization. The project is optimized to run on a GPU, requiring CUDA and cuDNN for efficient processing.

## Features

- **Transcription**: Uses `faster-whisper` and `ffmpeg` to transcribe audio from video files.
- **Summarization**: Utilizes `bart-large-cnn` to summarize the transcribed text.
- **FastAPI**: Provides an API for uploading videos and retrieving transcriptions and summaries.

## Requirements

- **CUDA**: Ensure that CUDA 12 is installed on your system ([cuda-toolkit](https://developer.nvidia.com/cuda-toolkit)).
- **cuDNN**: cuDNN 9 must be installed and properly configured ([cuDNN Installation](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/windows.html)).

## Installation

1. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

2. **Set up environment variables**:
    Create a `.env` file in the root directory and add the environment variable `CONFLUENCE_BASE_URL`.

## Usage

1. **Start the FastAPI server**:
    ```sh
    astapi dev api.py
    ```

2. **Access the API**:
    The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

- **POST /summarize**: Upload a video file and get the transcription and summary.
- **POST /post_to_confluence**: Create a Confluence page with the summarized content.
- **PUT /update_on_confluence**: Update an existing Confluence page with new content.