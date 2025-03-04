import logging
import os


def read_file(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf8") as file:
            return file.read()
    except FileNotFoundError as e:
        logging.error(f"{e}: File '{file_path}' not found.")
        return ""
    except Exception as e:
        logging.error(f"Error reading file: {e}")
        return ""


def write_to_file(file_path: str, content: str, mode: str = "w") -> None:
    try:
        with open(file_path, mode, encoding="utf8") as file:
            file.write(content)
    except Exception as e:
        logging.error(f"Error writing to file '{file_path}': {e}")


async def write_to_binary_file(file_path: str, uploaded_file) -> None:
    try:
        with open(file_path, "wb") as file:
            file.write(await uploaded_file.read())
    except Exception as e:
        logging.error(f"Error writing to file '{file_path}': {e}")


def create_output_folder(file_name: str) -> str:
    output_folder = os.path.join("output", file_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder
