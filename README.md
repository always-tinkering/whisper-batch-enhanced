# WhisperBatch

A robust tool for batch processing and transcribing audio and video files using OpenAI's Whisper model.

## Features

- Batch process multiple audio and video files
- Multi-threaded processing for improved performance
- Comprehensive error handling for CUDA/GPU issues
- Support for multiple output formats (SRT, VTT, TXT, JSON, TSV)
- Automatic device selection based on availability
- Detailed logging of transcription process

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/whisper-batch.git
cd whisper-batch

# Install the package
pip install -e .
```

## Usage

### Command Line Interface

```bash
# Basic usage
whisper-batch /path/to/videos /path/to/output

# Specify model size, device, and output format
whisper-batch /path/to/videos /path/to/output --model base.en --device auto --format srt

# Use multiple worker threads
whisper-batch /path/to/videos /path/to/output --workers 4

# Specify a different language
whisper-batch /path/to/videos /path/to/output --language fr
```

### Available Options

- `--model`: Model size to use (`tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3`)
- `--device`: Device to use for inference (`auto`, `cuda`, `cpu`)
- `--compute-type`: Compute type (`default`, `float16`, `int8`)
- `--format`: Output format (`srt`, `vtt`, `txt`, `json`, `tsv`)
- `--workers`: Number of concurrent workers (default: 1)
- `--language`: Language code for transcription (default: en)
- `--log-level`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

## CUDA/GPU Support and Troubleshooting

WhisperBatch includes enhanced error detection and handling for CUDA-related issues:

### Automatic DLL Checking

The system automatically checks for critical CUDA DLLs:
- `cudnn_ops64_9.dll`
- `cudnn64_8.dll`
- `cublas64_11.dll`

If these files are missing, the system will issue warnings and suggest solutions.

### Common CUDA Issues and Solutions

1. **Missing CUDA DLLs**: 
   - Download cuDNN from: https://developer.nvidia.com/cudnn
   - Extract and copy DLL files to your CUDA bin directory
   - Add CUDA bin directory to your PATH environment variable

2. **CUDA Out of Memory Errors**:
   - Try a smaller model (e.g., tiny.en or base.en)
   - Switch to CPU with `--device cpu`

3. **CUDNN Not Initialized**:
   - Install the matching cuDNN version for your CUDA installation
   - Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx

4. **"Can't find CUDA" or "CUDA driver version is insufficient"**:
   - Update your GPU drivers
   - Ensure CUDA Toolkit is properly installed

### Fallback to CPU

If CUDA errors are detected, WhisperBatch will automatically try to fall back to CPU processing when possible.

## Requirements

- Python 3.8+
- PyTorch
- OpenAI Whisper
- For GPU acceleration:
  - NVIDIA GPU with CUDA support
  - CUDA Toolkit
  - cuDNN

## How It Works

WhisperBatch processes audio/video files through these steps:
1. **File Discovery** - Recursively identifies all media files in the specified directory
2. **Audio Extraction** - Extracts audio from video files when necessary
3. **Parallel Processing** - Distributes transcription tasks across specified number of workers
4. **Transcription** - Uses OpenAI's Whisper model for high-quality speech recognition
5. **Result Generation** - Creates transcript files in your chosen format(s)

## Use Cases

- **Content Creators** - Generate subtitles for your videos automatically
- **Researchers** - Transcribe interviews and recordings for analysis
- **Media Companies** - Process large batches of media files efficiently
- **Archivists** - Convert speech to searchable text for digital archives
- **Language Learners** - Create study materials from audio/video content

## Supporting This Project

If you find WhisperBatch useful, please consider supporting its development:

- ☕ [Buy me a coffee](https://buymeacoffee.com/andrewgermann)
- ⭐ Star this repository on GitHub
- 🐛 Report bugs and suggest features via GitHub issues
- 🧪 Contribute code via pull requests

Your support helps maintain and improve this tool with new features and better performance!

## License

[MIT License](LICENSE) 