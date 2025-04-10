from faster_whisper import WhisperModel
from pathlib import Path
import logging
import torch # To check for CUDA availability
import platform
import os
import sys
import ctypes
import whisper

# Use shared logging configuration from the package
from whisper_batch import DEFAULT_LOG_FORMAT

# Configure basic logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)

# Global variable to hold the loaded model (to avoid reloading unnecessarily)
_model_cache = {}

def get_device(device_preference="auto"):
    """
    Determine the device to use for inference based on availability.
    
    Args:
        device_preference: One of "auto", "cuda", or "cpu".
    
    Returns:
        The device to use ("cuda" or "cpu").
    """
    if device_preference.lower() == "cpu":
        logging.info("Device preference is CPU")
        return "cpu"
    
    # Check if CUDA is available through PyTorch
    try:
        import torch
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            logging.info(f"CUDA is available with device: {torch.cuda.get_device_name(0)}")
            
            # Check for critical CUDA DLLs
            critical_dlls = ["cudnn_ops64_9.dll", "cudnn64_8.dll", "cublas64_11.dll"]
            dll_paths = [
                os.environ.get('CUDA_PATH', ''),
                os.path.join(os.environ.get('PROGRAMFILES', ''), 'NVIDIA GPU Computing Toolkit', 'CUDA'),
                os.getcwd(),
                os.path.dirname(os.path.abspath(__file__)),
            ]
            
            # Check if the critical DLLs are accessible
            missing_dlls = []
            for dll in critical_dlls:
                found = False
                for base_path in dll_paths:
                    if not base_path:
                        continue
                    
                    # Check bin directory and root directory
                    potential_paths = [
                        os.path.join(base_path, 'bin', dll),
                        os.path.join(base_path, dll)
                    ]
                    
                    for path in potential_paths:
                        if os.path.exists(path):
                            found = True
                            break
                    
                    if found:
                        break
                
                if not found:
                    missing_dlls.append(dll)
            
            if missing_dlls:
                dll_list = ", ".join(missing_dlls)
                logging.warning(f"CUDA is available but the following critical DLLs could not be found: {dll_list}")
                logging.warning("This may cause issues with GPU acceleration.")
                logging.warning("Solutions:")
                logging.warning("1. Install CUDA Toolkit and cuDNN: https://developer.nvidia.com/cuda-downloads")
                logging.warning("2. Add the CUDA bin directory to your PATH environment variable")
                logging.warning("3. Place the missing DLLs in your application directory")
                
                # If cudnn_ops64_9.dll is missing, suggest downloading cuDNN
                if "cudnn_ops64_9.dll" in missing_dlls:
                    logging.warning("cudnn_ops64_9.dll is missing - this is a common issue.")
                    logging.warning("Download cuDNN from: https://developer.nvidia.com/cudnn")
                    logging.warning("After downloading, extract and copy the DLL files to your CUDA bin directory")
                    
                # If device preference is explicitly CUDA, use it despite missing DLLs
                if device_preference.lower() == "cuda":
                    logging.warning("Proceeding with CUDA as explicitly requested, but errors may occur")
                    return "cuda"
                else:
                    logging.warning("Falling back to CPU due to missing critical DLLs")
                    return "cpu"
            
            # If device preference is "auto" or "cuda" and all checks passed, use CUDA
            if device_preference.lower() in ["auto", "cuda"]:
                return "cuda"
        else:
            logging.info("CUDA is not available through PyTorch")
            
            # If CUDA was explicitly requested but not available, log warning
            if device_preference.lower() == "cuda":
                logging.warning("CUDA was requested but is not available. Falling back to CPU.")
    except ImportError:
        logging.warning("PyTorch is not installed, cannot check CUDA availability")
    except Exception as e:
        logging.warning(f"Error checking CUDA availability: {str(e)}")
    
    # Default to CPU if CUDA is not available or there was an error
    return "cpu"

def load_transcription_model(model_size="base.en", device="auto", compute_type="default"):
    """
    Load the Whisper transcription model.
    
    Args:
        model_size: The size of the Whisper model to load.
        device: The device to use for inference (auto, cuda, or cpu).
        compute_type: The compute type to use (default, float16, int8).
    
    Returns:
        The loaded Whisper model, or None if loading failed.
    """
    try:
        # Determine the device
        actual_device = get_device(device)
        
        # Load the model
        logging.info(f"Loading Whisper model: {model_size} on {actual_device} with compute type {compute_type}")
        model = whisper.load_model(
            model_size,
            device=actual_device,
            download_root=None,
            in_memory=False,
        )
        
        return model
        
    except RuntimeError as e:
        error_message = str(e)
        
        # Check for common CUDA errors
        if "CUDA out of memory" in error_message:
            logging.error("CUDA out of memory error. Try a smaller model or use CPU.")
            logging.error("Available model sizes: tiny.en, base.en, small.en, medium.en")
            logging.error("You can switch to CPU with --device cpu")
        elif "CUDNN_STATUS_NOT_INITIALIZED" in error_message:
            logging.error("CUDNN not initialized properly. This might be due to incompatible CUDA versions.")
            logging.error("Try installing the matching CUDNN version for your CUDA installation.")
        elif "cudnn_ops64_9.dll" in error_message:
            logging.error("Missing cudnn_ops64_9.dll. This is a common issue with CUDA setup.")
            logging.error("Please download cuDNN from https://developer.nvidia.com/cudnn")
            logging.error("Extract the files and copy all .dll files to your CUDA bin directory")
            logging.error("You can also try running with --device cpu to use CPU instead")
        elif "Can't find CUDA" in error_message or "CUDA driver version is insufficient" in error_message:
            logging.error("CUDA driver issue. Your GPU drivers may need to be updated.")
            logging.error("Download latest NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx")
            logging.error("You can also try running with --device cpu to use CPU instead")
        elif "Cannot locate implementation" in error_message or "cannot find" in error_message.lower():
            logging.error("Missing CUDA or cuDNN libraries. Please check your CUDA installation.")
            logging.error("Make sure CUDA and cuDNN are properly installed and in your PATH")
        else:
            logging.error(f"Error loading Whisper model: {error_message}")
        
        # Attempt to fallback to CPU if CUDA was requested but failed
        if device in ["auto", "cuda"] and "cpu" != get_device("cpu"):
            logging.info("Attempting to fallback to CPU...")
            try:
                model = whisper.load_model(
                    model_size,
                    device="cpu",
                    download_root=None,
                    in_memory=False,
                )
                logging.info("Successfully loaded model on CPU instead")
                return model
            except Exception as cpu_error:
                logging.error(f"Failed to fallback to CPU: {str(cpu_error)}")
        
        return None
    
    except Exception as e:
        logging.error(f"Failed to load Whisper model: {str(e)}")
        return None

def is_media_file(file_path):
    """Check if a file is a media file that can be processed."""
    media_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.mp3', '.wav', '.m4a', '.flac']
    return Path(file_path).suffix.lower() in media_extensions

def get_media_files(directory_path):
    """Get all media files from a directory."""
    media_files = []
    dir_path = Path(directory_path)
    
    for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.mp3', '.wav', '.m4a', '.flac']:
        media_files.extend(dir_path.glob(f'**/*{ext}'))
    
    return media_files

def transcribe_file(model, file_path, output_file, output_format='srt', language='en'):
    """
    Transcribe a media file and save the result.
    
    Args:
        model: Loaded Whisper model
        file_path: Path to media file
        output_file: Path to save transcription
        output_format: Format to save (srt, vtt, txt, json, tsv)
        language: Language code for transcription
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create parent directory if it doesn't exist
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Transcribe the file
        logging.info(f"Transcribing {file_path}")
        result = model.transcribe(
            str(file_path),
            language=language,
            verbose=False
        )
        
        # Save the result in the specified format
        if output_format == 'srt':
            from whisper.utils import WriteSRT
            with open(output_file, 'w', encoding='utf-8') as f:
                writer = WriteSRT(output_file)
                writer.write_result(result)
        elif output_format == 'vtt':
            from whisper.utils import WriteVTT
            with open(output_file, 'w', encoding='utf-8') as f:
                writer = WriteVTT(output_file)
                writer.write_result(result)
        elif output_format == 'txt':
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result['text'])
        elif output_format == 'json':
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        elif output_format == 'tsv':
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("start\tend\ttext\n")
                for segment in result['segments']:
                    f.write(f"{segment['start']:.2f}\t{segment['end']:.2f}\t{segment['text']}\n")
        else:
            logging.error(f"Unsupported output format: {output_format}")
            return False
            
        logging.info(f"Transcription saved to {output_file}")
        return True
        
    except RuntimeError as e:
        error_message = str(e)
        
        # Handle specific CUDA errors during transcription
        if "CUDA out of memory" in error_message:
            logging.error(f"CUDA ran out of memory while transcribing {file_path}")
            logging.error("Try a smaller model or use CPU with --device cpu")
        elif "CUDNN_STATUS_NOT_INITIALIZED" in error_message:
            logging.error(f"CUDNN not initialized properly while transcribing {file_path}")
            logging.error("This might be due to incompatible CUDA versions")
        elif "cudnn_ops64_9.dll" in error_message:
            logging.error(f"Missing cudnn_ops64_9.dll while transcribing {file_path}")
            logging.error("Please download cuDNN from https://developer.nvidia.com/cudnn")
        else:
            logging.error(f"Runtime error transcribing {file_path}: {error_message}")
            
        return False
    except Exception as e:
        logging.error(f"Error transcribing {file_path}: {str(e)}")
        return False

if __name__ == '__main__':
    # Example usage for testing model loading
    print("\n--- Testing Model Loading ---")
    print("Attempting to load a tiny model (should be quick)...")
    # Using 'tiny.en' for faster loading during tests, 'auto' device detection
    loaded_model = load_transcription_model(model_size="tiny.en", device="auto")
    
    if loaded_model:
        print("Model loaded successfully.")
        # Test basic model information
        print(f"Device: {loaded_model.model.device}")
        print(f"Compute type: {loaded_model.model.compute_type}")
    else:
        print("Model loading failed.")
        print("Ensure faster-whisper is installed and has download permissions.") 