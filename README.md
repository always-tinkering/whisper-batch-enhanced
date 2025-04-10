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

## License

[MIT License](LICENSE) 