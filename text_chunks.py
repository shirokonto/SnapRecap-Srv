import logging
from typing import List

import nltk

nltk.download("punkt")


def split_text_into_chunks(document: str, max_tokens: int) -> List[str]:
    if not document:
        return []

    chunks, current_chunk, current_length = [], [], 0

    try:
        for sentence in nltk.sent_tokenize(document):
            sentence_length = len(sentence)  # Count characters

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
