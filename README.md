# Whisper Batch Enhanced

An enhanced version of the Whisper Batch tool for transcribing audio and video files. This version includes improved CUDA error handling and diagnostics to make GPU acceleration more robust.

## Features

- Batch processing of audio and video files
- Multi-format transcription output (SRT, VTT, TXT, JSON, TSV)
- Improved CUDA error handling and diagnostics
- Automatic fallback to CPU when CUDA issues are detected
- Detailed troubleshooting information for common CUDA/cuDNN problems
- Support for multiple languages
- Concurrent processing with configurable worker count

## Enhanced CUDA Error Handling

This enhanced version includes significant improvements for CUDA and cuDNN error detection and reporting:

- **Automatic DLL Detection**: Checks for critical CUDA DLLs including `cudnn_ops64_9.dll`, `cudnn64_8.dll`, and `cublas64_11.dll`
- **Multiple Path Scanning**: Searches multiple potential locations for required DLLs
- **Detailed Error Messages**: Provides specific error messages for common issues like "CUDA out of memory" and "CUDNN_STATUS_NOT_INITIALIZED"
- **Automatic Fallback**: Attempts to fall back to CPU processing when GPU issues are detected
- **Guided Troubleshooting**: Suggests specific solutions for each type of error

## Usage

```bash
python -m whisper_batch [input] [output] [options]
```

### Arguments

- `input`: Path to input directory or single file
- `output`: Path to output directory

### Options

- `--model`: Model size (tiny.en, base.en, small.en, medium.en, large-v3) [default: base.en]
- `--device`: Device to use (auto, cuda, cpu) [default: auto]
- `--compute-type`: Compute type (default, float16, int8) [default: default]
- `--format`: Output format (srt, vtt, txt, json, tsv) [default: srt]
- `--workers`: Number of concurrent workers [default: 1]
- `--language`: Language code [default: en]
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR) [default: INFO]

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Common CUDA Issues and Solutions

### Missing cudnn_ops64_9.dll

If you see an error about missing `cudnn_ops64_9.dll`:

1. Download cuDNN from https://developer.nvidia.com/cudnn (requires NVIDIA account)
2. Extract the files
3. Copy all DLL files to your CUDA bin directory (typically `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin\`)
4. Alternatively, run with `--device cpu` to use CPU instead

### CUDA Out of Memory

If you encounter "CUDA out of memory" errors:

1. Try a smaller model (e.g., tiny.en instead of medium.en)
2. Run with `--device cpu`
3. Close other GPU-intensive applications

### CUDNN Not Initialized

If CUDNN fails to initialize:

1. Ensure your CUDA toolkit and cuDNN versions are compatible
2. Re-install compatible versions
3. Add the CUDA bin directory to your PATH environment variable

## License

MIT

## Acknowledgments

This project is an enhanced version built on the Whisper model by OpenAI.