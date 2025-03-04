import logging
import os
from typing import List

import torch
from transformers import pipeline

from file_util import read_file, write_to_file
from text_chunks import split_text_into_chunks

# https://medium.com/towards-data-science/how-to-auto-generate-a-summary-from-long-youtube-videos-using-ai-a2a542b6698d

# Load summarization pipeline
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)


def summarize_chunks(chunks: List[str]) -> str:
    """
    Summarizes the input text chunks using the BART model.
    """
    summaries = []
    for idx, chunk in enumerate(chunks):
        try:
            summary = summarizer(chunk, max_length=124, min_length=30, do_sample=False)[
                0
            ]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            logging.error(f"Error summarizing chunk {idx}: {e}")
    print(f"Summaries: {summaries}")
    return " ".join(summaries)


def save_summary_to_file(summary: str, file_name: str, output_folder: str) -> None:
    try:
        base_name = os.path.basename(file_name)
        full_video_text_file = os.path.join(output_folder, f"summary-{base_name}")
        write_to_file(full_video_text_file, summary, mode="a")
    except Exception as e:
        logging.error(f"Error saving summary to file: {e}")


def summarize_whole(full_video_transcript_name, output_folder):
    """
    Summarizes the whole video transcript.
    """
    long_transcript = read_file(full_video_transcript_name)

    if not long_transcript:
        print("No content found in the transcription file.")
        return "No content to summarize."

    summary = None

    if long_transcript:
        # Split text into manageable chunks
        text_chunks = split_text_into_chunks(long_transcript, max_tokens=4000)
        summary = summarize_chunks(text_chunks)
        save_summary_to_file(
            summary, f"summary_{full_video_transcript_name}", output_folder
        )

        if len(summary) > 5000:
            # If the summary is too long we can reapply the summarization function
            text_chunks = split_text_into_chunks(summary, max_tokens=1000)
            short_summary = summarize_chunks(text_chunks)
            summary = f"Section: Video Summary\nSummary: {short_summary}"
            save_summary_to_file(
                short_summary,
                f"short_summary_{full_video_transcript_name}",
                output_folder,
            )
            logging.info("Summary saved to file.")
    else:
        logging.error("Error: Unable to process the text.")

    return f"Section: Video Summary\nSummary: {summary}"
