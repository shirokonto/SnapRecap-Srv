import torch
from transformers import pipeline

# Load summarization pipeline
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
# summarizer = pipeline("summarization", model="t5-base", device=device)


def summarize_section(section):
    """
    Summarizes the content of a given section using summarization pipeline.
    """
    content = section["content"]

    # Skip empty content
    if not content:
        return {"title": section["title"], "summary": "No content to summarize."}

    try:
        summary = summarizer(content, max_length=130, min_length=30, do_sample=False)[
            0
        ]["summary_text"]

    except Exception as e:
        summary = f"Error summarizing section: {str(e)}"
    return {"title": section["title"], "summary": summary}


def sort_text_to_section_headers(transcription_chunks, section_titles):

    sorted_sections = [{"title": title, "content": ""} for title in section_titles]
    current_section_index = 0

    for chunk in transcription_chunks:
        text = chunk["text"].strip()
        if not text:
            continue

        # If there is a next section, check if this chunk signals its start.
        if current_section_index < len(section_titles) - 1:
            next_section_marker = section_titles[current_section_index + 1].lower()
            if next_section_marker in text.lower():
                current_section_index += 1

        sorted_sections[current_section_index]["content"] += text + " "

    print(f"Sorted Sections: {sorted_sections}")
    return sorted_sections


def process_transcription(transcription_chunks, section_titles):
    """
    Main function to process the transcription file.
    Either summarize the whole video
    or detects sections, splits the text and summarizes each section.
    """

    sorted_sections = sort_text_to_section_headers(transcription_chunks, section_titles)
    print("Assigned Sections:")
    for sec in sorted_sections:
        print(
            f"Title: {sec['title']}, Content length: {len(sec['content'])} characters"
        )

    summarized_sections = [summarize_section(sec) for sec in sorted_sections]
    # Combine all summaries
    summary = "\n\n".join(
        f"Section: {s['title']}\nSummary: {s['summary']}" for s in summarized_sections
    )

    print(f"End result: {summary}")
    return summary


# Example usage
if __name__ == "__main__":
    transcription_test_file = "example_transcription.srt"
    sct_titles = []
    process_transcription(transcription_test_file, sct_titles)
