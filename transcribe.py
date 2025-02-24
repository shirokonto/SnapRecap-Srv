import ctypes
import math
import os
import re

import ffmpeg
from faster_whisper import WhisperModel


def check_cudnn_dll():
    """
    Checks if cuDNN DLL is loaded
    """
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


def extract_audio(input_video, output_folder: str) -> str:
    """
    Extracts audio from a given video file and saves it as a .wav file in output.
    :param input_video: Video file path
    :param output_folder: Folder to save the audio file
    :return extracted_audio: Path to the extracted audio file
    """

    print(f"Extracting audio from {input_video}")
    input_video_name = input_video.replace(".mp4", "")
    extracted_audio = os.path.join(output_folder, f"audio-{input_video_name}.wav")
    stream = ffmpeg.input(input_video)
    stream = ffmpeg.output(stream, extracted_audio)
    ffmpeg.run(stream, overwrite_output=True)
    return extracted_audio


def transcribe(audio):
    """
    Transcribes audio file using Whisper model.
    :param audio:
    :return: detected language and segments [] (containing id, start, end and text)
    """
    model = WhisperModel("small", device="cuda")
    segments, info = model.transcribe(audio)
    language = info.language if hasattr(info, "language") else "unknown"

    return language, segments


def format_time(seconds):
    """
    Converts start and end time of segments to HH:MM:SS,sss format
    :param seconds: SS.ss
    :return: HH:MM:SS, sss
    """
    hours = math.floor(seconds / 3600)
    seconds %= 3600
    minutes = math.floor(seconds / 60)
    seconds %= 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"

    return formatted_time


def generate_subtitle_file(
    input_video_name, output_folder, language, segments, sections=None
):
    """
    Generates and saves a subtitle file(s) in SRT format from the transcription segments.
    If sections are given, only the subtitle file is saved.
    Else, both subtitle and full video text files are saved.
    :param input_video_name: Name of the video
    :param output_folder: Folder to save the subtitle files
    :param language: Detected language
    :param segments: Transcription segments
    :param sections: Section titles (if any)
    :return: Path to the saved subtitle file(s)
    """
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
        subtitle_text += f"{segment_start} --> {segment_end}\n"
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
    with open(transcription_file, "r") as f:
        subtitle_content = f.read()

    # Regex to extract timestamps and text
    pattern = r"(\d+)\s+([\d:,]+) --> ([\d:,]+)\s+(.*?)(?=\n\d+\s+[\d:,]+ -->|\Z)"
    matches = re.findall(pattern, subtitle_content, re.DOTALL)

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


def generate_transcription(input_video, output_folder, sections=None):
    """
    Main function to handle video transcription and subtitle file generation.
    """
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
    print(
        generate_transcription(
            input_test_video, os.path.join("output", "input_test.mp4")
        )
    )
