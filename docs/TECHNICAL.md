# WhisperBatch Technical Documentation

This document provides technical details about the WhisperBatch architecture, code organization, and implementation details to help developers understand and extend the codebase.

## Architecture Overview

WhisperBatch is organized as a Python package with several key components:

```
whisper-batch/
├── src/
│   ├── whisper_batch/       # Core package
│   │   ├── __init__.py      # Package initialization
│   │   ├── main.py          # CLI entry point
│   │   ├── transcriber.py   # Core transcription logic
│   │   ├── file_handler.py  # File operations
│   │   └── audio_extractor.py # Audio extraction
│   ├── gui/
│   │   └── tkinter_app.py   # GUI application
│   └── core/
│       └── transcription_processor.py # Lower-level transcription logic
├── docs/                    # Documentation
├── setup.py                 # Package setup
└── README.md                # Main README
```

### Component Responsibilities

1. **transcriber.py**: Responsible for:
   - CUDA/GPU error detection and handling
   - Loading and managing Whisper models
   - Transcribing media files
   - DLL dependency checking
   - Automatic device selection

2. **main.py**: Provides:
   - Command-line interface
   - Batch processing coordination
   - Multi-threading management
   - Progress tracking and reporting

3. **tkinter_app.py**: Implements:
   - Graphical user interface
   - User-friendly controls
   - Visual progress indication
   - Interactive error handling

4. **transcription_processor.py**: Handles:
   - Low-level transcription operations
   - Audio processing
   - Model interaction
   - Format conversion

## Code Flow

### Transcription Process Flow

1. **Input Processing**: 
   - Validate input paths
   - Discover media files
   - Create output directory structure

2. **Model Loading**:
   - Check device availability (CUDA/CPU)
   - Verify DLL dependencies for GPU acceleration
   - Load appropriate Whisper model
   - Fall back to CPU if GPU issues are detected

3. **Batch Processing**:
   - Create worker pool based on `--workers` parameter
   - Distribute files among workers
   - Track progress across all workers

4. **Transcription**:
   - Process each file with the Whisper model
   - Handle transcription results
   - Format output according to chosen format
   - Save to output location

5. **Error Handling**:
   - Detect CUDA/GPU errors
   - Provide specific error messages
   - Attempt recovery by falling back to CPU
   - Log detailed error information

## Key Implementation Details

### CUDA Error Handling

The CUDA error handling in `transcriber.py` is implemented through multiple layers:

1. **DLL Detection**:
   ```python
   # Check for critical CUDA DLLs
   critical_dlls = ["cudnn_ops64_9.dll", "cudnn64_8.dll", "cublas64_11.dll"]
   dll_paths = [
       os.environ.get('CUDA_PATH', ''),
       os.path.join(os.environ.get('PROGRAMFILES', ''), 'NVIDIA GPU Computing Toolkit', 'CUDA'),
       os.getcwd(),
       os.path.dirname(os.path.abspath(__file__)),
   ]
   ```

2. **Error Pattern Matching**:
   ```python
   # Check for common CUDA errors
   if "CUDA out of memory" in error_message:
       logging.error("CUDA out of memory error. Try a smaller model or use CPU.")
   elif "CUDNN_STATUS_NOT_INITIALIZED" in error_message:
       logging.error("CUDNN not initialized properly. This might be due to incompatible CUDA versions.")
   ```

3. **CPU Fallback Mechanism**:
   ```python
   # Attempt to fallback to CPU if CUDA was requested but failed
   if device in ["auto", "cuda"] and "cpu" != get_device("cpu"):
       logging.info("Attempting to fallback to CPU...")
       try:
           model = whisper.load_model(model_size, device="cpu")
           return model
       except Exception as cpu_error:
           logging.error(f"Failed to fallback to CPU: {str(cpu_error)}")
   ```

### Multi-threading Implementation

The multi-threading system in `main.py` uses Python's `concurrent.futures` module:

```python
# Use ThreadPoolExecutor for concurrent processing
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    # Submit all tasks
    future_to_file = {executor.submit(process_single_video, file): file for file in video_files}
    
    # Process as they complete
    for future in as_completed(future_to_file):
        result = future.result()
        results.append(result)
```

This approach allows for:
- Parallel processing of multiple files
- Dynamic completion handling
- Proper error propagation
- Resource management

## Extending the Codebase

### Adding a New Output Format

To add a new output format (e.g., "csv"):

1. Update the CLI argument choices in `main.py`:
   ```python
   parser.add_argument("--format", help="Output format", 
                      choices=["srt", "vtt", "txt", "json", "tsv", "csv"], 
                      default="srt")
   ```

2. Add the format handling in `transcribe_file()` function in `transcriber.py`:
   ```python
   elif output_format == 'csv':
       with open(output_file, 'w', encoding='utf-8') as f:
           f.write("start,end,text\n")
           for segment in result['segments']:
               f.write(f"{segment['start']:.2f},{segment['end']:.2f},{segment['text']}\n")
   ```

### Supporting a New Model Type

To add support for a new Whisper model variant:

1. Update the model choices in `main.py`:
   ```python
   parser.add_argument("--model", help="Model size", 
                      choices=["tiny.en", "base.en", "small.en", "medium.en", "large-v3", "new-model"], 
                      default="base.en")
   ```

2. Handle any special loading requirements in `load_transcription_model()` in `transcriber.py`.

## Performance Considerations

1. **Memory Usage**:
   - The Whisper models have varying memory requirements:
     - tiny: ~75MB
     - base: ~150MB
     - small: ~500MB
     - medium: ~1.5GB
     - large: ~3GB
   - When using GPU acceleration, ensure sufficient VRAM is available

2. **CPU vs. GPU Performance**:
   - GPU processing can be 5-10x faster than CPU
   - For small files (<1 minute), the overhead of loading the model may outweigh the performance benefit
   - Multiple workers (2-3) are optimal for GPU to balance between I/O and compute time

3. **Worker Count Optimization**:
   - For CPU: Set workers to the number of available CPU cores
   - For GPU: Use 2-3 workers to keep the GPU fed while handling I/O

## Logging System

The application uses Python's standard logging system with enhanced formatting:

```python
logging.basicConfig(
    level=getattr(logging, args.log_level),
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
```

Log levels:
- DEBUG: Detailed information for diagnostics
- INFO: Confirmation of expected behavior
- WARNING: Indication of potential issues
- ERROR: Error conditions preventing functionality

## Testing

To run tests:

```bash
# From the project root
pytest tests/
```

Key test areas:
1. CUDA detection and fallback
2. Model loading with various configurations
3. File processing and format conversion
4. Error handling and recovery

## Common Development Tasks

### Adding a New Command-line Option

1. Add the argument to the parser in `main.py`:
   ```python
   parser.add_argument("--new-option", help="Description", type=str, default="default_value")
   ```

2. Pass the new option to the `process_videos()` function:
   ```python
   results = process_videos(
       # ... existing arguments
       new_option=args.new_option
   )
   ```

3. Update the function signature and implementation to use the new option.

### Improving GPU Error Detection

To add detection for a new type of GPU error:

1. Identify the error pattern in the exception message
2. Add a new condition in the error handling section of `load_transcription_model()`:
   ```python
   elif "new error pattern" in error_message:
       logging.error("New error detected. Here's how to fix it.")
       logging.error("Additional information and solutions.")
   ```

## Environment Variables

WhisperBatch respects the following environment variables:

- `CUDA_PATH`: Path to CUDA installation (for DLL detection)
- `WHISPER_BATCH_LOG_LEVEL`: Default logging level if not specified in CLI
- `WHISPER_BATCH_MODEL_DIR`: Custom directory for downloading/loading models 