import math
import ffmpeg
import os
import ctypes
from faster_whisper import WhisperModel

input_video = ""

# Function to check if cuDNN DLL is loaded
def check_cudnn_dll():
    dll_ops_path = os.path.join(os.environ['CUDA_PATH'], 'bin', 'cudnn_ops_infer64_8.dll')
    dll_cnn_path = os.path.join(os.environ['CUDA_PATH'], 'bin', 'cudnn_cnn_infer64_8.dll')

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

# Function to extract audio from a given video file
def extract_audio(input_video):
    global input_video_name

    input_video_name = input_video.replace(".mp4", "")
    extracted_audio = f"audio-{input_video_name}.wav"
    stream = ffmpeg.input(input_video)
    stream = ffmpeg.output(stream, extracted_audio)
    ffmpeg.run(stream, overwrite_output=True)
    return extracted_audio

# Function to transcribe audio using Whisper
def transcribe(audio):
    model = WhisperModel("small", device="cuda")
    result = model.transcribe(audio)

    # Check if result contains both segments and language as separate elements.
    segments = list(result[0])  # The segments are usually the first element.
    info = result[1]  # Info (which contains the language) is the second element if available.

    # Make sure to extract the language correctly.
    language = info.get('language', 'unknown') if isinstance(info, dict) else "unknown"

    transcription = "\n".join(
        f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}" for segment in segments
    )
    return language, segments

    #model = WhisperModel("small", device="cuda")
    #segments, info = model.transcribe(audio)
    #language = info[0]
    #print("Transcription language:", language)
    #segments = list(segments)
    #transcription = "\n".join(
    #    f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}" for segment in segments
    #)
    # not returning correct issue
    #return transcription

# converting transcription segments start and end time displayed as
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
def generate_subtitle_file(language, segments):
    global input_video_name

    #     language, segments = transcribe(audio=audio_file)
    input_video_name = input_video.replace(".mp4", "")

    subtitle_file = f"sub-{input_video_name}.{language}.srt"
    text = ""
    for index, segment in enumerate(segments):
        segment_start = format_time(segment.start)
        segment_end = format_time(segment.end)
        text += f"{str(index + 1)} \n"
        text += f"{segment_start} --> {segment_end} \n"
        text += f"{segment.text} \n"
        text += "\n"

    f = open(subtitle_file, "w")
    f.write(text)
    f.close()

    return subtitle_file

# Main function to handle video transcription
def process_video(input_video):

    check_cudnn_dll()

    audio_file = extract_audio(input_video)

    language, segments = transcribe(audio=audio_file)
    # ValueError: too many values to unpack (expected 2)

    subtitle_file = generate_subtitle_file(
        language=language,
        segments=segments
    )
    return subtitle_file

if __name__ == "__main__":
    input_video = "input.mp4"
    print(process_video(input_video))
