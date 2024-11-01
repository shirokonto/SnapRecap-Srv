import torch
from transformers import pipeline

# Loads summarization pipeline
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)


def summarize_srt(file_path):
    with open(file_path, "r") as file:
        content = file.readlines()

    # TODO adapt later to include timestamps
    # Removes timestamps and numbers and combines lines into text block
    subtitle_text = " ".join(
        [
            line.strip()
            for line in content
            if not line.strip().isdigit() and "-->" not in line
        ]
    )

    # Prepares input prompt for summarization
    prompt = (
        f"Rewrite the following text in a formal, third-person style, avoiding any use of 'you' or 'I' and use "
        f"an impersonal and professional tone, divide the following text into meaningful sections, and provide a formal "
        f"and concise summary for each section :\n\n{subtitle_text}"
    )
    """prompt = (
        f"Summarize the following content by dividing it into meaningful sections. "
        f"For each section, provide a clear heading and a concise summary in a formal and impersonal tone:\n\n{subtitle_text}"
    )"""

    # Splits content in chunks since BART models maximal input size is 1024
    max_chunk_size = 1024  # BART's max input size
    chunks = [
        prompt[i : i + max_chunk_size] for i in range(0, len(prompt), max_chunk_size)
    ]

    # TODO gets weird summary
    # Generates summary for each chunk and combines them
    summary = ""
    for chunk in chunks:
        result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summary += result[0]["summary_text"] + " "
    # summary = summarizer(prompt, max_length=150, min_length=30, do_sample=False)

    print(f"Summary: {summary}")

    return summary
