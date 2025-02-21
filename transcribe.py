import ctypes
import math
import os
import re

import ffmpeg
from faster_whisper import WhisperModel


# Checks if cuDNN DLL is loaded
def check_cudnn_dll():
    dll_ops_path = os.path.join(
        os.environ["CUDA_PATH"], "bin", "cudnn_ops_infer64_8.dll"
    )
    dll_cnn_path = os.path.join(
        os.environ["CUDA_PATH"], "bin", "cudnn_cnn_infer64_8.dll"
    )

    try:
        ctypes.WinDLL(dll_ops_path)
        print(f"Successfully loaded cuDNN DLL: {dll_ops_path}")
    except Exception as e:
        print(f"Failed to load cuDNN DLL: {e}")

    try:
        ctypes.WinDLL(dll_cnn_path)
        print(f"Successfully loaded cuDNN DLL: {dll_cnn_path}")
    except Exception as e:
        print(f"Failed to load cuDNN DLL: {e}")


# Extracts audio from a given video file
def extract_audio(input_video, output_folder):
    print(f"Extracting audio from {input_video}")
    input_video_name = input_video.replace(".mp4", "")
    extracted_audio = os.path.join(output_folder, f"audio-{input_video_name}.wav")
    stream = ffmpeg.input(input_video)
    stream = ffmpeg.output(stream, extracted_audio)
    ffmpeg.run(stream, overwrite_output=True)
    return extracted_audio


# Transcribes audio using Whisper
def transcribe(audio):
    model = WhisperModel("small", device="cuda")
    segments, info = model.transcribe(audio)
    language = info.language if hasattr(info, "language") else "unknown"

    return language, segments


# Converts transcription segments start and end time displayed as
# 00:00:10,500 --> 00:00:15,000  in seconds to HH:MM:SS, sss
def format_time(seconds):
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"

    return formatted_time


# Takes the language detected of the audio and transcription segments
# and creates a subtitle file in SRT form
def generate_subtitle_file(
    input_video_name, output_folder, language, segments, sections=None
):
    input_video_name = input_video_name.replace("temp_", "")
    subtitle_file = os.path.join(
        output_folder, f"sub-{input_video_name}.{language}.srt"
    )

    subtitle_text = ""
    full_video_text = ""

    for index, segment in enumerate(segments):
        segment_start = format_time(segment.start)
        segment_end = format_time(segment.end)
        subtitle_text += f"{str(index + 1)} \n"
        subtitle_text += f"{segment_start} --> {segment_end}\n{segment.text}\n\n"
        subtitle_text += f"{segment.text} \n\n"
        if sections is None:
            full_video_text += f"{segment.text} \n"

    with open(subtitle_file, "w", encoding="utf-8") as f:
        f.write(subtitle_text)
        f.close()

    full_video_text_file = None
    if sections is None:
        full_video_text_file = os.path.join(
            output_folder, f"only-text-{input_video_name}.{language}.srt"
        )
        with open(full_video_text_file, "w", encoding="utf-8") as f:
            f.write(full_video_text)
            f.close()

    return (subtitle_file, full_video_text_file) if sections is None else subtitle_file


def split_transcription(transcription_file):
    # Regex to extract timestamps and text
    pattern = r"(\d+)\s+([\d:,]+) --> ([\d:,]+)\s+(.*?)(?=\n\d+\s+[\d:,]+ -->|\Z)"
    matches = re.findall(pattern, transcription_file, re.DOTALL)

    transcription_chunks = []
    for match in matches:
        index, start_time, end_time, text = match
        transcription_chunks.append(
            {
                "index": int(index),
                "start_time": start_time.strip(),
                "end_time": end_time.strip(),
                "text": text.strip().replace("\n", " "),
            }
        )

    return transcription_chunks


# Main function to handle video transcription
def process_video(input_video, output_folder, sections=None):
    check_cudnn_dll()

    input_video_name = os.path.splitext(os.path.basename(input_video))[0]

    audio_file = extract_audio(input_video, output_folder)
    language, segments = transcribe(audio=audio_file)

    result = generate_subtitle_file(
        input_video_name=input_video_name,
        output_folder=output_folder,
        language=language,
        segments=segments,
        sections=sections,
    )

    # If sections are given, return only the subtitle file else both subtitle and full video text file
    if isinstance(result, tuple):
        subtitle_file, full_video_text_file = result
        print(
            f"Subtitle file saved at: {subtitle_file} and full video at: {full_video_text_file}"
        )
        return subtitle_file, full_video_text_file
    else:
        subtitle_file = result
        print(f"Subtitle file saved at: {subtitle_file}")
        return subtitle_file


if __name__ == "__main__":
    input_test_video = "input.mp4"
    print(process_video(input_test_video, os.path.join("output", "input_test.mp4")))
