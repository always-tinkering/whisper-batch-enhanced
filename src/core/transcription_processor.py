import logging
import os
import time
from pathlib import Path
from queue import Queue
import torch
import whisper
from typing import List, Tuple, Dict, Optional, Union, Any

# Import for CUDA error detection
from whisper.audio import SAMPLE_RATE

logger = logging.getLogger(__name__)

class TranscriptionProcessor:
    """Processor class for handling audio file transcriptions using Whisper."""
    
    def __init__(self, model_name="tiny", device=None, progress_queue=None, output_format="txt"):
        """
        Initialize the transcription processor.
        
        Args:
            model_name (str): The Whisper model to use ('tiny', 'base', 'small', 'medium', 'large')
            device (str): Device to use ('cuda', 'cpu', or None for auto-detection)
            progress_queue (Queue): Queue for reporting progress
            output_format (str): Output format ('txt', 'srt', 'vtt', 'json')
        """
        self.model_name = model_name
        self.requested_device = device
        self.device = self._determine_device(device)
        self.progress_queue = progress_queue
        self.output_format = output_format
        self.model = None
        
    def _determine_device(self, device):
        """
        Determine which device to use for processing.
        
        Args:
            device (str): Requested device ('cuda', 'cpu', or None for auto-detection)
            
        Returns:
            str: The device to use ('cuda' or 'cpu')
        """
        # If specific device requested, try to use it
        if device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available. Falling back to CPU.")
                return "cpu"
            
            # Check CUDA version compatibility
            try:
                cuda_version = torch.version.cuda
                logger.info(f"CUDA version: {cuda_version}")
                return "cuda"
            except Exception as e:
                logger.warning(f"Error checking CUDA version: {e}. Falling back to CPU.")
                return "cpu"
                
        elif device == "cpu":
            return "cpu"
            
        # Auto-detection (device=None)
        try:
            if torch.cuda.is_available():
                logger.info(f"CUDA detected. Using GPU: {torch.cuda.get_device_name(0)}")
                return "cuda"
            else:
                logger.info("No CUDA device available. Using CPU.")
                return "cpu"
        except Exception as e:
            logger.warning(f"Error during CUDA detection: {e}. Using CPU instead.")
            return "cpu"
    
    def load_model(self):
        """Load the Whisper model."""
        if self.model:
            return
            
        logger.info(f"Loading Whisper {self.model_name} model...")
        start_time = time.time()
        
        try:
            # Try loading with requested device
            self.model = whisper.load_model(self.model_name, device=self.device)
            load_time = time.time() - start_time
            logger.info(f"Model loaded on {self.device} in {load_time:.2f} seconds.")
            
        except Exception as e:
            # If loading on GPU failed and it wasn't explicitly CPU only, try CPU
            if self.device == "cuda" and self.requested_device != "cpu":
                logger.warning(f"Failed to load model on GPU: {e}")
                logger.info("Attempting to load model on CPU instead...")
                
                try:
                    # Update device to CPU
                    self.device = "cpu"
                    self.model = whisper.load_model(self.model_name, device="cpu")
                    load_time = time.time() - start_time
                    logger.info(f"Model loaded on CPU in {load_time:.2f} seconds.")
                    
                except Exception as cpu_error:
                    logger.error(f"Failed to load model on CPU: {cpu_error}")
                    raise RuntimeError(f"Could not load Whisper model on GPU or CPU: {str(e)}\nCPU Error: {str(cpu_error)}")
            else:
                # If CPU was explicitly requested or some other error occurred
                logger.error(f"Failed to load model: {e}")
                raise
    
    def process_files(self, files):
        """
        Process a list of audio files for transcription.
        
        Args:
            files (list): List of Path objects for audio files
        """
        # Load the model
        try:
            self.load_model()
        except Exception as e:
            if self.progress_queue:
                self.progress_queue.put(("error", str(e)))
            logger.error(f"Failed to load model: {e}")
            return
        
        # Process files
        total_files = len(files)
        
        for i, file in enumerate(files, 1):
            # Report progress
            if self.progress_queue:
                self.progress_queue.put((i, total_files, file.name))
            
            logger.info(f"Processing {i}/{total_files}: {file.name}")
            
            try:
                self._process_single_file(file)
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error processing {file.name}: {error_msg}")
                
                # Check if this is a CUDA error and we're not already on CPU
                if self.device == "cuda" and any(cuda_term in error_msg.lower() for cuda_term in 
                                               ["cuda", "cudnn", "gpu", "nvrtc", "cublas"]):
                    logger.warning(f"CUDA error detected. Attempting to fall back to CPU for {file.name}")
                    
                    try:
                        # Switch to CPU for this file
                        old_device = self.device
                        self.device = "cpu"
                        
                        # If needed, reload model on CPU
                        if self.model and hasattr(self.model, "device") and str(self.model.device) != "cpu":
                            logger.info("Reloading model on CPU...")
                            self.model = whisper.load_model(self.model_name, device="cpu")
                        
                        # Retry processing
                        logger.info(f"Retrying {file.name} on CPU")
                        self._process_single_file(file)
                        
                        # Keep using CPU for remaining files if GPU failed
                        logger.info("Continuing with CPU for remaining files")
                        
                    except Exception as cpu_error:
                        logger.error(f"CPU fallback also failed for {file.name}: {cpu_error}")
                        if self.progress_queue:
                            self.progress_queue.put(("error", f"Transcription failed on both GPU and CPU: {cpu_error}"))
                        return
                else:
                    # Not a CUDA error or already on CPU, so report and continue
                    if self.progress_queue:
                        self.progress_queue.put(("error", f"Error processing {file.name}: {error_msg}"))
                    return
        
        # Report completion
        if self.progress_queue:
            self.progress_queue.put(("complete", None))
            
    def _process_single_file(self, file):
        """
        Process a single audio file.
        
        Args:
            file (Path): Path object for the audio file
        """
        # Check if file exists
        if not file.exists():
            raise FileNotFoundError(f"Audio file not found: {file}")
            
        # Check if output file already exists
        output_path = file.with_suffix(f".{self.output_format}")
        if output_path.exists():
            logger.info(f"Output file already exists: {output_path}. Skipping.")
            return
        
        # Transcribe the audio file
        try:
            logger.info(f"Transcribing {file} using {self.device}...")
            start_time = time.time()
            
            # Load audio and pad/trim it
            audio = whisper.load_audio(str(file))
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.device)
            
            # Detect the spoken language
            _, probs = self.model.detect_language(mel)
            detected_language = max(probs, key=probs.get)
            logger.info(f"Detected language: {detected_language}")
            
            # Decode the audio
            options = {
                "fp16": self.device == "cuda"
            }
            result = self.model.transcribe(str(file), **options)
            
            # Save the transcription
            self._save_transcription(file, result)
            
            # Log completion
            elapsed_time = time.time() - start_time
            logger.info(f"Transcription complete in {elapsed_time:.2f} seconds.")
            
        except Exception as e:
            logger.error(f"Error transcribing {file}: {e}")
            raise
    
    def _save_transcription(self, file, result):
        """
        Save the transcription results to a file.
        
        Args:
            file (Path): Path object for the audio file
            result (dict): Whisper transcription result
        """
        # Determine the output file path based on the original file path
        output_path = file.with_suffix(f".{self.output_format}")
        
        # Save in the requested format
        if self.output_format == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result["text"])
                
        elif self.output_format == "json":
            import json
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
        elif self.output_format in ["srt", "vtt"]:
            from whisper.utils import get_writer
            writer = get_writer(self.output_format, output_dir=file.parent)
            writer(result, file.stem)
            
        else:
            logger.warning(f"Unsupported output format: {self.output_format}. Using txt instead.")
            with open(file.with_suffix(".txt"), "w", encoding="utf-8") as f:
                f.write(result["text"])
                
        logger.info(f"Saved transcription to {output_path}") 