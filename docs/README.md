# WhisperBatch

A comprehensive tool for batch processing and transcribing audio and video files using OpenAI's Whisper speech recognition models. WhisperBatch provides robust error handling, particularly for CUDA and GPU-related issues, making it more reliable in production environments.

![WhisperBatch Logo](https://via.placeholder.com/800x400?text=WhisperBatch)

## Features

- **Batch Processing**: Process multiple media files in one command
- **Multi-threaded Processing**: Concurrent transcription for improved performance
- **Comprehensive Error Handling**: 
  - Detailed CUDA/GPU error detection and recovery
  - Automatic DLL dependency checking
  - Smart fallback to CPU when GPU issues occur
- **Flexible Output Formats**: 
  - SRT and VTT subtitles
  - Plain text transcriptions
  - JSON for structured data
  - TSV for tabular format
- **User Interfaces**:
  - Command-line interface for scripting and automation
  - Graphical user interface for desktop use
- **Automatic Device Selection**: Optimally chooses between GPU and CPU
- **Deep Logging**: Detailed logging for troubleshooting

## Installation

### Prerequisites

- Python 3.8 or higher
- FFmpeg (for audio extraction from video files)
- For GPU acceleration:
  - NVIDIA GPU with CUDA support
  - CUDA Toolkit 11.x or higher
  - cuDNN (matching your CUDA version)

### Method 1: Install from PyPI (Recommended)

```bash
pip install whisper-batch
```

### Method 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/whisper-batch.git
cd whisper-batch

# Install the package in development mode
pip install -e .
```

### GPU Acceleration Setup

For optimal performance with GPU acceleration:

1. Install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
2. Install [cuDNN](https://developer.nvidia.com/cudnn)
3. Ensure the following DLLs are in your PATH:
   - `cudnn_ops64_9.dll`
   - `cudnn64_8.dll`
   - `cublas64_11.dll`

## Usage

### Command Line Interface

The command line interface provides full access to all features:

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

| Option | Description | Default | Available Values |
|--------|-------------|---------|------------------|
| `--model` | Whisper model size | `base.en` | `tiny.en`, `base.en`, `small.en`, `medium.en`, `large-v3` |
| `--device` | Computation device | `auto` | `auto`, `cuda`, `cpu` |
| `--compute-type` | Compute precision | `default` | `default`, `float16`, `int8` |
| `--format` | Output format | `srt` | `srt`, `vtt`, `txt`, `json`, `tsv` |
| `--workers` | Concurrent workers | `1` | Integer value |
| `--language` | Language code | `en` | ISO language codes |
| `--log-level` | Logging verbosity | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Graphical User Interface

To launch the GUI application:

```bash
whisper-batch-gui
```

The GUI provides access to all features through a user-friendly interface.

## CUDA/GPU Support and Troubleshooting

WhisperBatch includes enhanced error detection and handling for CUDA-related issues.

### Automatic DLL Checking

The system automatically checks for critical CUDA DLLs and provides warnings if they're missing:
- `cudnn_ops64_9.dll`
- `cudnn64_8.dll`
- `cublas64_11.dll`

### Common CUDA Issues and Solutions

#### 1. Missing CUDA DLLs
- **Error**: `cudnn_ops64_9.dll not found` or similar messages
- **Solution**: 
  - Download cuDNN from: https://developer.nvidia.com/cudnn
  - Extract and copy DLL files to your CUDA bin directory
  - Add CUDA bin directory to your PATH environment variable

#### 2. CUDA Out of Memory Errors
- **Error**: `CUDA out of memory` or `CUDA error: out of memory`
- **Solution**:
  - Try a smaller model (e.g., tiny.en or base.en)
  - Reduce batch size if applicable
  - Close other GPU-intensive applications
  - Switch to CPU with `--device cpu`

#### 3. CUDNN Not Initialized
- **Error**: `CUDNN_STATUS_NOT_INITIALIZED` or `Failed to initialize cuDNN`
- **Solution**:
  - Install the matching cuDNN version for your CUDA installation
  - Update NVIDIA drivers: https://www.nvidia.com/Download/index.aspx
  - Ensure PATH includes cuDNN directories

#### 4. CUDA Driver Issues
- **Error**: `Can't find CUDA` or `CUDA driver version is insufficient`
- **Solution**:
  - Update your GPU drivers to the latest version
  - Ensure CUDA Toolkit is properly installed
  - Verify compatibility between your GPU, driver, and CUDA version

### Fallback to CPU

If CUDA errors are detected, WhisperBatch will automatically try to fall back to CPU processing when possible. This ensures transcription jobs complete even when GPU issues occur.

## Python API

WhisperBatch can also be used as a Python library in your own projects:

```python
from whisper_batch.main import process_videos

# Process videos with custom settings
results = process_videos(
    input_path="path/to/videos",
    output_path="path/to/output",
    model_size="base.en",
    device="auto",
    compute_type="default",
    output_format="srt",
    max_workers=4,
    language="en"
)

# Check results
for file_path, success, error_msg in results:
    if success:
        print(f"Successfully processed: {file_path}")
    else:
        print(f"Failed to process: {file_path} - {error_msg}")
```

## Performance Tips

1. **Model Selection**: Smaller models (`tiny.en`, `base.en`) are much faster but less accurate
2. **GPU Acceleration**: Using CUDA with a compatible GPU can provide 5-10x speedup
3. **Concurrent Workers**: Set `--workers` to the number of CPU cores for CPU mode, or 2-3 for GPU mode
4. **File Size**: Split very large audio files (>30 minutes) for better memory management

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 