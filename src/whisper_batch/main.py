import argparse
import logging
from pathlib import Path
import os
import sys
from tqdm import tqdm
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union, List, Tuple
import time

# Ensure the src directory is in the Python path
# This allows importing modules from whisper_batch
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    # Use shared logging configuration from the package
    from whisper_batch import DEFAULT_LOG_FORMAT
    from whisper_batch.file_handler import find_video_files, SUPPORTED_VIDEO_EXTENSIONS
    from whisper_batch.audio_extractor import extract_audio
    from whisper_batch.transcriber import load_transcription_model, transcribe_file, is_media_file, get_media_files
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure you are running from the project root or the script can find the 'src' directory.")
    sys.exit(1)

# Configure basic logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)

def create_output_path(input_video_path: Path, input_base_dir: Path, output_base_dir: Path, output_format: str) -> Path:
    """
    Determines the output path for the transcript file, mirroring the input structure.

    Args:
        input_video_path: Path to the original video file.
        input_base_dir: The root directory where the search started.
        output_base_dir: The root directory where transcripts should be saved.
        output_format: The desired output file extension (e.g., "txt", "srt").

    Returns:
        The calculated Path object for the output transcript file.
    """
    # Get the relative path of the video file with respect to the input base directory
    relative_path = input_video_path.relative_to(input_base_dir)
    # Construct the output path by joining the output base dir with the relative path
    output_path = output_base_dir / relative_path
    # Change the file extension to the desired output format
    output_file_path = output_path.with_suffix(f".{output_format}")
    return output_file_path

def format_timestamp(seconds: float, always_include_hours: bool = False) -> str:
    """
    Format seconds into an SRT timestamp (HH:MM:SS,mmm).
    
    Args:
        seconds: The time in seconds.
        always_include_hours: Whether to always include the hours part.
    
    Returns:
        A string formatted as an SRT timestamp.
    """
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int(seconds * 1000) % 1000
    
    if always_include_hours or hours > 0:
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    else:
        return f"{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def format_srt(segments) -> str:
    """
    Format transcription segments into SRT subtitle format.
    
    Args:
        segments: Iterable of segment objects from faster_whisper.
    
    Returns:
        A string containing the SRT formatted subtitles.
    """
    srt_content = []
    for i, segment in enumerate(segments, start=1):
        # Format: subtitle number
        srt_content.append(str(i))
        
        # Format: start time --> end time
        start_time = format_timestamp(segment.start, always_include_hours=True)
        end_time = format_timestamp(segment.end, always_include_hours=True)
        srt_content.append(f"{start_time} --> {end_time}")
        
        # Format: subtitle text
        srt_content.append(segment.text.strip())
        
        # Add an empty line between entries
        srt_content.append("")
    
    return "\n".join(srt_content)

def process_videos(
    input_path: Union[str, Path], 
    output_path: Union[str, Path], 
    model_size: str = "base.en", 
    device: str = "auto", 
    compute_type: str = "default", 
    output_format: str = "srt", 
    max_workers: int = 1,
    language: str = "en"
) -> List[Tuple[str, bool, str]]:
    """
    Process videos in the input path and generate transcriptions.
    
    Args:
        input_path: Path to input directory or single file
        output_path: Path to output directory
        model_size: Size of the Whisper model to use
        device: Device to use for inference (auto, cuda, cpu)
        compute_type: Compute type to use
        output_format: Output format for transcriptions
        max_workers: Maximum number of concurrent workers
        language: Language code for transcription
    
    Returns:
        A list of tuples containing (file_path, success, error_message)
    """
    # Convert paths to Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Load the transcription model
    logging.info(f"Loading transcription model: {model_size} on {device}")
    model = load_transcription_model(model_size, device, compute_type)
    
    if model is None:
        logging.error("Failed to load transcription model. Cannot proceed with transcription.")
        return []
    
    # Create output directory if it doesn't exist
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect video files to process
    video_files = []
    if input_path.is_file():
        if is_media_file(input_path):
            video_files = [input_path]
        else:
            logging.warning(f"The file {input_path} is not recognized as a media file.")
            return []
    else:
        video_files = get_media_files(input_path)
        if not video_files:
            logging.warning(f"No media files found in {input_path}")
            return []
    
    logging.info(f"Found {len(video_files)} media files to process")
    
    # Process videos
    results = []
    start_time = time.time()
    
    # Function to process a single video
    def process_single_video(video_file):
        try:
            rel_path = video_file.relative_to(input_path) if input_path.is_dir() else video_file.name
            output_file = output_path / rel_path.with_suffix(f".{output_format}")
            
            # Create subdirectories in output path if needed
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            logging.info(f"Processing {video_file}")
            success = transcribe_file(
                model, 
                video_file, 
                output_file, 
                output_format=output_format,
                language=language
            )
            
            if success:
                return (str(video_file), True, "")
            else:
                return (str(video_file), False, "Transcription failed")
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error processing {video_file}: {error_msg}")
            return (str(video_file), False, error_msg)
    
    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_video, file): file for file in video_files}
        
        # Process as they complete
        for future in as_completed(future_to_file):
            result = future.result()
            results.append(result)
            file_path, success, error_msg = result
            
            if success:
                logging.info(f"Successfully processed: {file_path}")
            else:
                logging.error(f"Failed to process: {file_path} - {error_msg}")
    
    # Summarize results
    elapsed_time = time.time() - start_time
    success_count = sum(1 for _, success, _ in results if success)
    
    logging.info(f"Processing complete. Processed {len(results)} files in {elapsed_time:.2f} seconds.")
    logging.info(f"Success: {success_count}, Failed: {len(results) - success_count}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Batch process videos with Whisper speech recognition")
    parser.add_argument("input", help="Input directory or file", type=str)
    parser.add_argument("output", help="Output directory", type=str)
    parser.add_argument("--model", help="Model size", choices=["tiny.en", "base.en", "small.en", "medium.en", "large-v3"], default="base.en")
    parser.add_argument("--device", help="Device to use", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--compute-type", help="Compute type", choices=["default", "float16", "int8"], default="default")
    parser.add_argument("--format", help="Output format", choices=["srt", "vtt", "txt", "json", "tsv"], default="srt")
    parser.add_argument("--workers", help="Number of concurrent workers", type=int, default=1)
    parser.add_argument("--language", help="Language code", type=str, default="en")
    parser.add_argument("--log-level", help="Logging level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    try:
        # Process videos
        results = process_videos(
            args.input,
            args.output,
            args.model,
            args.device,
            args.compute_type,
            args.format,
            args.workers,
            args.language
        )
        
        # Print summary
        success_count = sum(1 for _, success, _ in results if success)
        if results:
            print(f"\nSummary: Successfully processed {success_count} out of {len(results)} files.")
            
            if success_count < len(results):
                print("\nFailed files:")
                for file_path, success, error_msg in results:
                    if not success:
                        print(f"  - {file_path}: {error_msg}")
        else:
            print("\nNo files were processed. Please check the input path and file types.")
            
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        import traceback
        logging.debug(traceback.format_exc())
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 