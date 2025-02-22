import logging
from typing import List

import nltk

nltk.download("punkt")


def read_file(file_name: str) -> str:
    try:
        with open(file_name, "r", encoding="utf8") as file:
            return file.read()
    except FileNotFoundError as e:
        logging.error(f"{e}: File '{file_name}' not found.")
        return ""
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return ""


def split_text_into_chunks(document: str, max_tokens: int) -> List[str]:
    if not document:
        return []

    chunks, current_chunk, current_length = [], [], 0

    try:
        for sentence in nltk.sent_tokenize(document):
            sentence_length = len(sentence.split())  # Count tokens (words)

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


# long_text = read_file(VIDEO_NAME)
# if long_text:
#    text_chunks = split_text_into_chunks(long_text, max_tokens=4000)
#    logging.info(f"Text chunks: {text_chunks}")
# else:
#    logging.error("Error: Unable to process the text.")
