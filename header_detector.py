def detect_section_headers(transcription_chunks, sections):
    section_mapping = {section: [] for section in sections}

    # Convert section titles for case-insensitive matching
    lower_sections = [s.lower() for s in sections]

    current_section_idx = 0

    for chunk in transcription_chunks:
        idx = chunk["index"]
        text = chunk["text"].lower()

        # Check if current chunk contains next section header
        if current_section_idx < len(sections) - 1:
            next_section_title = lower_sections[current_section_idx + 1]
            if next_section_title in text:
                current_section_idx += 1

        # Assign chunk to current section
        section_mapping[sections[current_section_idx]].append(idx)

    return section_mapping
