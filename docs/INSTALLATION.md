# WhisperBatch Installation Guide

This guide provides step-by-step instructions for installing WhisperBatch on Windows, macOS, and Linux.

## Prerequisites

Before installing WhisperBatch, you need:

1. **Python 3.8 or higher** installed on your system
2. **FFmpeg** for audio extraction from video files
3. **pip** (Python package manager)
4. For GPU acceleration:
   - NVIDIA GPU with CUDA support
   - CUDA Toolkit and cuDNN installed

## Windows Installation

### Step 1: Install Python
1. Download Python 3.8 or higher from [python.org](https://www.python.org/downloads/windows/)
2. Run the installer, select "Add Python to PATH"
3. Click "Install Now"
4. Verify installation by opening Command Prompt and typing:
   ```
   python --version
   ```

### Step 2: Install FFmpeg
1. Download FFmpeg from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip)
2. Extract the ZIP file
3. Copy the `bin` folder contents to a permanent location (e.g., `C:\ffmpeg\bin`)
4. Add to PATH:
   - Right-click on "This PC" → Properties
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Add the bin directory (e.g., `C:\ffmpeg\bin`)
   - Click OK on all dialogs
5. Verify installation:
   ```
   ffmpeg -version
   ```

### Step 3: Install WhisperBatch
1. Open Command Prompt
2. Install the package:
   ```
   pip install whisper-batch
   ```
   
   OR for development installation:
   ```
   git clone https://github.com/yourusername/whisper-batch.git
   cd whisper-batch
   pip install -e .
   ```

### Step 4: GPU Acceleration Setup (Optional)
1. Install NVIDIA GPU drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx)
2. Install CUDA Toolkit 11.x from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
3. Install cuDNN (requires NVIDIA developer account):
   - Download from [developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
   - Extract and copy files to your CUDA directory (see [CUDA_TROUBLESHOOTING.md](CUDA_TROUBLESHOOTING.md))

## macOS Installation

### Step 1: Install Python
1. Install Homebrew if not installed:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install Python:
   ```bash
   brew install python@3.10
   ```
3. Verify installation:
   ```bash
   python3 --version
   ```

### Step 2: Install FFmpeg
```bash
brew install ffmpeg
```

### Step 3: Install WhisperBatch
```bash
pip3 install whisper-batch
```

OR for development installation:
```bash
git clone https://github.com/yourusername/whisper-batch.git
cd whisper-batch
pip3 install -e .
```

### Step 4: GPU Acceleration
Note: GPU acceleration is not available on macOS for CUDA. WhisperBatch will automatically use CPU mode.

## Linux Installation

### Step 1: Install Python
For Ubuntu/Debian:
```bash
sudo apt update
sudo apt install python3 python3-pip
```

For Fedora:
```bash
sudo dnf install python3 python3-pip
```

### Step 2: Install FFmpeg
For Ubuntu/Debian:
```bash
sudo apt install ffmpeg
```

For Fedora:
```bash
sudo dnf install ffmpeg
```

### Step 3: Install WhisperBatch
```bash
pip3 install whisper-batch
```

OR for development installation:
```bash
git clone https://github.com/yourusername/whisper-batch.git
cd whisper-batch
pip3 install -e .
```

### Step 4: GPU Acceleration Setup (Optional)
1. Install NVIDIA GPU drivers:
   ```bash
   # Ubuntu/Debian
   sudo apt install nvidia-driver-525  # Or newer
   
   # Fedora
   sudo dnf install akmod-nvidia
   ```

2. Install CUDA Toolkit:
   ```bash
   # Ubuntu/Debian (example for CUDA 11.8)
   wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
   sudo sh cuda_11.8.0_520.61.05_linux.run
   ```

3. Install cuDNN (after downloading from NVIDIA with developer account):
   ```bash
   # Extract and copy
   tar -xf cudnn-linux-x86_64-*.tar.xz
   sudo cp cudnn-linux-*/include/cudnn*.h /usr/local/cuda/include
   sudo cp cudnn-linux-*/lib/libcudnn* /usr/local/cuda/lib64
   sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
   ```

4. Add to PATH:
   ```bash
   echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
   echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
   source ~/.bashrc
   ```

## Verifying Installation

Test your installation by running:

```bash
# Display help
whisper-batch --help

# Run a simple test on a short video
whisper-batch /path/to/video/file.mp4 /path/to/output
```

## Upgrading WhisperBatch

To upgrade to the latest version:

```bash
pip install --upgrade whisper-batch
```

## Troubleshooting

If you encounter issues:

1. Check that all prerequisites are installed correctly
2. Verify Python and FFmpeg are in your PATH
3. For CUDA issues, see [CUDA_TROUBLESHOOTING.md](CUDA_TROUBLESHOOTING.md)
4. Try running with --device cpu to bypass GPU issues

For more help, create an issue on GitHub. 