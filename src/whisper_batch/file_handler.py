import os
from pathlib import Path
import logging

# Use shared logging configuration from the package
from whisper_batch import DEFAULT_LOG_FORMAT

# Configure basic logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)

# Define supported video file extensions (add more as needed)
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv"}

def find_video_files(input_dir: Path) -> list[Path]:
    """
    Recursively finds all video files in the input directory.

    Args:
        input_dir: The Path object representing the directory to scan.

    Returns:
        A list of Path objects for all found video files.
    """
    video_files = []
    if not input_dir.is_dir():
        logging.error(f"Input path is not a valid directory: {input_dir}")
        return []

    logging.info(f"Scanning for video files in: {input_dir}")
    for item in input_dir.rglob('*'): # rglob searches recursively
        if item.is_file() and item.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            video_files.append(item)
            logging.debug(f"Found video file: {item}")

    if not video_files:
        logging.warning(f"No video files found in {input_dir}")
    else:
        logging.info(f"Found {len(video_files)} video file(s).")

    return video_files

if __name__ == '__main__':
    # Example usage for testing
    test_dir = Path('.') # Replace with a directory containing test videos
    if test_dir.is_dir():
        found_files = find_video_files(test_dir)
        print("\n--- Found Files ---")
        if found_files:
            for file_path in found_files:
                print(file_path)
        else:
            print("No video files were found.")
    else:
        print(f"Test directory not found: {test_dir}")