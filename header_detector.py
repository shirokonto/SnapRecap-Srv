def detect_section_headers(transcription_chunks, sections):
    if not sections:
        return detect_section_headers_without_sections(transcription_chunks)
    else:
        return detect_section_headers_with_sections(transcription_chunks, sections)


def detect_section_headers_without_sections(transcription_chunks):
    print(f"reaches: detect_section_headers_without_sections")
    # TODO implement when no sections are provided
    return {}


def detect_section_headers_with_sections(transcription_chunks, sections):
    section_mapping = {section: [] for section in sections}

    # Assigns chunks sequentially to sections
    current_section_idx = 0

    for chunk in transcription_chunks:
        idx = chunk["index"]

        # If it moves past the current section, switch to the next section
        if current_section_idx < len(sections) - 1:
            next_section_start = [
                i
                for i, c in enumerate(transcription_chunks)
                if sections[current_section_idx + 1].lower() in c["text"].lower()
            ]
            if next_section_start and idx >= next_section_start[0]:
                current_section_idx += 1

        # Assigns the chunk to the current section
        section_mapping[sections[current_section_idx]].append(idx)

    return section_mapping
