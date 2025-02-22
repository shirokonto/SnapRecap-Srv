import logging
from typing import List

import nltk
import torch
from transformers import pipeline

# Download punkt tokenizer
nltk.download("punkt")
nltk.download("punkt_tab")

# Load summarization pipeline
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

# BART summarizer parameters
bart_params = {
    "max_length": 124,
    "min_length": 30,
    "do_sample": False,
    "truncation": True,
    "repetition_penalty": 2.0,
}


def recursive_summarization(summary: str, max_tokens=1024):
    """
    Re-applies summarization if the summary exceeds token limits.
    """
    if len(summary.split()) <= max_tokens:
        return summary

    chunks = split_text_into_chunks(summary, max_tokens=max_tokens)
    logging.info(f"Re-summarizing into smaller chunks: {len(chunks)}")

    return get_summary_bart(chunks)


def get_summary_bart(list_chunks: List[str]) -> str:
    """
    Summarizes the input text chunks using the BART model.
    """
    summaries = []
    for idx, chunk in enumerate(list_chunks):
        chunk = chunk.strip()

        # Skip empty chunks
        if not chunk:
            logging.warning(f"Skipped empty chunk at index {idx}.")
            continue

        try:
            # Truncate chunk if too long (BART usually supports ~1024 tokens)
            if len(chunk.split()) > 1024:
                logging.warning(
                    f"Chunk at index {idx} exceeds 1024 tokens. Truncating."
                )
                chunk = " ".join(chunk.split()[:1024])

            summary = summarizer(chunk, **bart_params)[0]["summary_text"]
            summaries.append(summary)
        except Exception as e:
            logging.error(f"Error summarizing chunk at index {idx}: {e}")

    if summaries:
        return " ".join(summaries)
    else:
        logging.error("No summaries were generated.")
        return ""


def detect_section_headers_one(transcription_file):
    # TODO replace with header_detector
    with open(
        transcription_file, "r"
    ) as file:  # TODO dont read just pass subtitle_content?
        lines = file.readlines()

    sections = []
    current_section = {"title": None, "content": ""}

    # Keywords indicate new sections
    section_keywords = [
        "What is",
        "How does",
        "Introduction",
        "Overview",
        "Conclusion",
        "conclusion",
        "Summary",
        "Getting started",
    ]
    # TODO if sections are given use them
    # TODO if keywords are given use them - else keyword extraction and search for similar ones
    # TODO exclude stuff like "What is up"

    # Process each line, skip timestamp lines
    for line in lines:
        stripped_line = line.strip()
        if "-->" in stripped_line:
            continue

        # Check if line matches any section keyword
        if any(keyword in stripped_line for keyword in section_keywords):
            # Save the current section if it exists
            if current_section["title"]:
                sections.append(current_section)

            # Start a new section
            current_section = {"title": stripped_line, "content": ""}

        # Add content to the current section
        elif current_section["title"]:
            current_section["content"] += stripped_line + " "

    # Append the last section
    if current_section["title"]:
        sections.append(current_section)

    return sections


def split_text_into_chunks(document: str, max_tokens: int) -> list:
    """
    Splits the input document into chunks with a maximum token limit.
    """
    if not document:
        return []

    chunks, current_chunk, current_length = [], [], 0

    try:
        for sentence in nltk.sent_tokenize(document):
            sentence_length = len(sentence.split())

            # If sentence is too long, truncate
            if sentence_length > max_tokens:
                logging.warning("A sentence exceeds max_tokens. Truncating.")
                sentence = " ".join(sentence.split()[:max_tokens])
                sentence_length = max_tokens

            if current_length + sentence_length < max_tokens:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [sentence], sentence_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    except Exception as e:
        logging.error(f"Error splitting text into chunks: {e}")
        return []


def summarize_chunks(chunks, summarizer, summarization_params):
    """
    Applies summarization to each chunk and combines results.
    """
    try:
        summaries = [
            summarizer(chunk, **summarization_params)[0]["summary_text"]
            for chunk in chunks
        ]
        return " ".join(summaries)
    except Exception as e:
        logging.error(f"Error summarizing chunks: {e}")
        return ""


def summarize_section(section):
    """
    Summarizes the content of a given section using summarization pipeline.
    """
    content = section["content"].strip()

    # Skip empty content
    if not content:
        return {"title": section["title"], "summary": "No content to summarize."}

    try:
        chunks = split_text_into_chunks(content, max_tokens=4000)
        summary = summarize_chunks(chunks, summarizer, bart_params)

        # If summary is still too long, re-summarize
        if len(summary.split()) > 5000:
            smaller_chunks = split_text_into_chunks(summary, max_tokens=1000)
            summary = summarize_chunks(smaller_chunks, summarizer, bart_params)

    except Exception as e:
        summary = f"Error summarizing section: {str(e)}"
    return {"title": section["title"], "summary": summary}


def process_transcription(transcription_file, sections=None):
    """
    Main function to process the transcription file.
    Either summarize the whole video
    or detects sections, splits the text and summarizes each section.
    """
    if sections is None:
        # Summarize the whole video
        with open(transcription_file, "r") as file:
            content = file.read()

        if not content:
            print("No content found in the transcription file.")
            return "No content to summarize."

        # Split text into chunks and summarize
        chunks = split_text_into_chunks(content, max_tokens=4000)
        summarized_video = summarize_chunks(chunks, summarizer, bart_params)

        # Re-summarize if needed
        if len(summarized_video.split()) > 5000:
            smaller_chunks = split_text_into_chunks(summarized_video, max_tokens=1000)
            summarized_video = summarize_chunks(smaller_chunks, summarizer, bart_params)

        summary = f"Section: Video Summary\nSummary: {summarized_video}"
        # summarized_video = summarize_section(
        #    {"title": "Video Summary", "content": content}
        # )
        # summary = f"Section: {summarized_video['title']}\nSummary: {summarized_video['summary']}"

    else:
        # Detect sections in the transcription
        # sections = detect_section_headers(transcription_file)
        sections = detect_section_headers_one(transcription_file)

        print(f"Results - Sections: {sections}")

        # Summarize each section
        summarized_sections = []
        for section in sections:
            summarized_section = summarize_section(section)
            summarized_sections.append(summarized_section)

        # Combine all summaries
        summary = "\n\n".join(
            f"Section: {s['title']}\nSummary: {s['summary']}"
            for s in summarized_sections
        )

    print(f"END result: {summary}")
    return summary


# Example usage
if __name__ == "__main__":
    transcription_test_file = "example_transcription.srt"
    process_transcription(transcription_test_file)
