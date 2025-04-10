import ffmpeg
from pathlib import Path
import logging
import tempfile
import os
import shutil

# Use shared logging configuration from the package
from whisper_batch import DEFAULT_LOG_FORMAT

# Configure basic logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)

def extract_audio(video_path: Path, output_dir: Path | None = None) -> Path | None:
    """
    Extracts audio from a video file using FFmpeg and saves it as a WAV file.

    Args:
        video_path: Path to the input video file.
        output_dir: Optional directory to save the audio file. If None,
                    a temporary directory is used.

    Returns:
        The Path object of the extracted audio file (WAV format),
        or None if extraction fails.
    """
    # Check if FFmpeg is installed and available
    if not shutil.which("ffmpeg"):
        logging.error("FFmpeg is not installed or not in the system PATH.")
        logging.error("Please install FFmpeg (https://ffmpeg.org/download.html) and make sure it's in your PATH.")
        return None

    if not video_path.is_file():
        logging.error(f"Video file not found: {video_path}")
        return None

    try:
        # Determine output path
        if output_dir:
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            # Use video filename but change extension to .wav
            audio_filename = video_path.stem + ".wav"
            audio_output_path = output_dir / audio_filename
        else:
            # Create a temporary file for the audio output
            # Use suffix ".wav" to ensure correct format handling later
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            audio_output_path = Path(temp_file.name)
            temp_file.close() # Close the file handle, but the file remains

        logging.info(f"Extracting audio from '{video_path.name}' to '{audio_output_path}'...")

        # FFmpeg command based on product doc recommendation
        (
            ffmpeg
            .input(str(video_path))
            .output(
                str(audio_output_path),
                ac=1,          # Mono channel
                ar=16000,      # 16kHz sample rate
                vn=None,       # No video output
                acodec='pcm_s16le' # Standard WAV codec
            )
            .overwrite_output() # Overwrite if exists (useful for reruns/temp files)
            .run(capture_stdout=True, capture_stderr=True, quiet=True) # Use quiet=True to suppress ffmpeg console output
        )

        logging.info(f"Successfully extracted audio to: {audio_output_path}")
        return audio_output_path

    except ffmpeg.Error as e:
        logging.error(f"FFmpeg error extracting audio from {video_path}:")
        try:
            # Decode stderr for more detailed error messages
            error_message = e.stderr.decode('utf-8')
            logging.error(f"FFmpeg stderr: {error_message}")
        except Exception:
            logging.error(f"FFmpeg stderr could not be decoded: {e.stderr}")
        # Clean up temporary file if it exists and an error occurred
        if not output_dir and 'audio_output_path' in locals() and audio_output_path.exists():
             try:
                 os.remove(audio_output_path)
                 logging.debug(f"Cleaned up temporary audio file: {audio_output_path}")
             except OSError as rm_err:
                 logging.error(f"Failed to remove temporary audio file {audio_output_path}: {rm_err}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error extracting audio from {video_path}: {e}")
        # Clean up temporary file if it exists and an error occurred
        if not output_dir and 'audio_output_path' in locals() and audio_output_path.exists():
             try:
                 os.remove(audio_output_path)
                 logging.debug(f"Cleaned up temporary audio file: {audio_output_path}")
             except OSError as rm_err:
                 logging.error(f"Failed to remove temporary audio file {audio_output_path}: {rm_err}")
        return None


if __name__ == '__main__':
    # Example Usage (requires a dummy video file named 'test_video.mp4')
    # Create a dummy file for testing if it doesn't exist
    test_video = Path("test_video.mp4")
    if not test_video.exists():
        print("Creating dummy 'test_video.mp4' for testing...")
        # You might need ffmpeg installed to create a valid dummy,
        # or just create an empty file. For simplicity, creating empty.
        test_video.touch()
        print(f"Note: Created an empty file. Real extraction requires a valid video.")

    if test_video.is_file():
        print(f"\n--- Testing Audio Extraction (using '{test_video}') ---")
        # Test with temporary file output
        print("\nTesting with temporary output:")
        extracted_audio_temp = extract_audio(test_video)
        if extracted_audio_temp:
            print(f"Audio extracted to temporary file: {extracted_audio_temp}")
            print(f"Temporary file exists: {extracted_audio_temp.exists()}")
            # Clean up the temp file after test
            try:
                os.remove(extracted_audio_temp)
                print(f"Cleaned up: {extracted_audio_temp}")
            except OSError as e:
                print(f"Error removing temp file {extracted_audio_temp}: {e}")
        else:
            print("Audio extraction failed (temporary output). Check logs.")

        # Test with specified output directory
        print("\nTesting with specified output directory:")
        output_audio_dir = Path("./temp_audio_output")
        extracted_audio_specific = extract_audio(test_video, output_dir=output_audio_dir)
        if extracted_audio_specific:
            print(f"Audio extracted to specific directory: {extracted_audio_specific}")
            print(f"Specific output file exists: {extracted_audio_specific.exists()}")
            # Optionally clean up the specific output dir after test
            # import shutil
            # if output_audio_dir.exists():
            #     shutil.rmtree(output_audio_dir)
            #     print(f"Cleaned up directory: {output_audio_dir}")
        else:
            print("Audio extraction failed (specific output). Check logs.")
    else:
        print(f"Test video file '{test_video}' not found.") 