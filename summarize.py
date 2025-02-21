import torch
from transformers import pipeline

# Load summarization pipeline
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)


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


def summarize_section(section):
    """
    Summarizes the content of a given section using summarization pipeline.
    """
    content = section["content"]
    try:
        summary = summarizer(content, max_length=130, min_length=30, do_sample=False)[
            0
        ]["summary_text"]
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
        summarized_video = summarize_section(
            {"title": "Video Summary", "content": content}
        )
        summary = f"Section: {summarized_video['title']}\nSummary: {summarized_video['summary']}"

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
