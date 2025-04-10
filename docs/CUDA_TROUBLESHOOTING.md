# CUDA Troubleshooting Guide for WhisperBatch

This guide provides detailed solutions for CUDA-related issues when using WhisperBatch with GPU acceleration.

## CUDA Setup Requirements

For optimal GPU-accelerated transcription, you need:

1. **NVIDIA GPU**: CUDA-compatible GPU (GTX 1060 or newer recommended)
2. **GPU Drivers**: Latest NVIDIA drivers for your GPU
3. **CUDA Toolkit**: CUDA 11.x or newer
4. **cuDNN Library**: Version compatible with your CUDA installation

## Common Issues and Solutions

### 1. "cudnn_ops64_9.dll not found" or Missing DLL Errors

This is one of the most common issues with CUDA-accelerated applications.

#### Symptoms:
- Error message mentions missing DLL files 
- Application falls back to CPU
- Slow transcription performance

#### Solutions:

**Step 1: Install cuDNN properly**
1. Download cuDNN from [NVIDIA's cuDNN page](https://developer.nvidia.com/cudnn) (requires NVIDIA developer account)
2. Extract the ZIP file
3. Copy these files from the extracted folder to your CUDA installation:
   - Copy `bin/*.dll` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin`
   - Copy `include/*.h` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\include`
   - Copy `lib/*.lib` to `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\lib\x64`

**Step 2: Ensure PATH environment variable includes CUDA bin directory**
1. Right-click on "This PC" or "My Computer" → Properties
2. Click "Advanced system settings"
3. Click "Environment Variables"
4. Under "System variables", find "Path" and click "Edit"
5. Add the CUDA bin directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin`)
6. Click OK on all dialogs

**Step 3: Check DLL files directly**
1. Navigate to your CUDA bin directory
2. Verify these files exist:
   - `cudnn_ops64_9.dll`
   - `cudnn64_8.dll`
   - `cublas64_11.dll`
3. If missing, re-install cuDNN and copy files as in Step 1

**Step 4: Alternative approach for temporary fix**
If you continue having issues, you can place the required DLL files directly in:
1. Your WhisperBatch installation directory
2. Your current working directory where you run the command
3. Your system32 folder (not recommended but works as a last resort)

### 2. "CUDA out of memory" Errors

These errors occur when your GPU doesn't have enough VRAM for the model.

#### Symptoms:
- Error messages containing "CUDA out of memory"
- Transcription begins but crashes
- Only occurs with larger models or longer audio files

#### Solutions:

**Step 1: Use a smaller model**
```bash
whisper-batch /path/to/videos /path/to/output --model tiny.en --device cuda
```

|  Model   | Approx. VRAM Required |
|----------|------------------------|
| tiny.en  | 1 GB                  |
| base.en  | 1.5 GB                |
| small.en | 2.5 GB                |
| medium.en| 5 GB                  |
| large-v3 | 10+ GB                |

**Step 2: Process shorter files**
Split longer audio files into smaller segments (5-10 minutes each).

**Step 3: Close other GPU applications**
- Check Task Manager for other applications using the GPU
- Close web browsers (especially if playing videos)
- Close other AI or rendering applications

**Step 4: Manage memory settings**
If you're familiar with PyTorch, you can modify `setup.py` to add:
```python
import torch
torch.cuda.empty_cache()  # Add this before processing each file
```

**Step 5: Use CPU as fallback**
If GPU memory issues persist:
```bash
whisper-batch /path/to/videos /path/to/output --device cpu
```

### 3. "CUDNN_STATUS_NOT_INITIALIZED" Errors

These errors typically indicate compatibility issues between CUDA components.

#### Symptoms:
- Error message includes "CUDNN_STATUS_NOT_INITIALIZED"
- Application crashes when initializing the model
- GPU is detected but can't be used

#### Solutions:

**Step 1: Verify CUDA version compatibility**
Ensure your CUDA version matches the cuDNN version:

|  CUDA Version  | Compatible cuDNN |
|----------------|------------------|
| CUDA 11.0-11.1 | cuDNN 8.0.x      |
| CUDA 11.2-11.3 | cuDNN 8.1.x      |
| CUDA 11.4-11.6 | cuDNN 8.2.x      |
| CUDA 11.7+     | cuDNN 8.4.x      |

**Step 2: Reinstall matching versions**
1. Uninstall current CUDA and cuDNN
2. Download CUDA from [NVIDIA's CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive)
3. Download matching cuDNN from [NVIDIA's cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive)
4. Install CUDA first, then cuDNN

**Step 3: Check PyTorch CUDA compatibility**
Ensure your PyTorch version is compatible with your CUDA version:

```bash
# For CUDA 11.7
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu117

# For CUDA 11.8
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118
```

**Step 4: Reinstall PyTorch with CUDA support**
```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. "CUDA driver version is insufficient" Errors

This error occurs when your NVIDIA driver is too old for your CUDA version.

#### Symptoms:
- Error mentions "CUDA driver version is insufficient"
- Error states "CUDA capability X.X required, device has X.X"
- GPU is detected but can't be used

#### Solutions:

**Step 1: Check driver and CUDA compatibility**
Each CUDA version requires a minimum driver version:

| CUDA Version | Minimum Driver Version |
|--------------|------------------------|
| CUDA 11.0    | 450.36.06              |
| CUDA 11.1    | 455.23                 |
| CUDA 11.2    | 460.27.03              |
| CUDA 11.3    | 465.19.01              |
| CUDA 11.4    | 470.42.01              |
| CUDA 11.5    | 495.29.05              |
| CUDA 11.6    | 510.39.01              |
| CUDA 11.7    | 515.43.04              |
| CUDA 11.8    | 520.61.05              |

**Step 2: Update GPU drivers**
1. Visit [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx)
2. Select your GPU model and operating system
3. Download and install the latest driver
4. Restart your computer

**Step 3: Verify GPU compatibility**
Check if your GPU supports the CUDA version you're trying to use. Older GPUs may not support newer CUDA versions.

### 5. WhisperBatch Automatically Falls Back to CPU

WhisperBatch is designed to automatically fall back to CPU if it detects CUDA issues.

#### Symptoms:
- Log message: "Falling back to CPU due to missing critical DLLs"
- Log message: "Attempting to fallback to CPU..."
- Transcription runs but is slower than expected

#### Understanding the fallback mechanism:

The fallback is triggered by:
1. Missing CUDA DLLs
2. CUDA initialization errors
3. GPU memory errors

To disable automatic fallback and force errors:
```bash
whisper-batch /path/to/videos /path/to/output --device cuda
```

Using `--device cuda` explicitly will prevent CPU fallback and show the actual error.

## Advanced Troubleshooting

### Running a CUDA Test Script

Create and run this diagnostic script to check your CUDA setup:

```python
# cuda_test.py
import torch
import sys

def test_cuda():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            print(f"CUDA device {i}: {device_name} (Compute Capability: {device_capability})")
            
        # Test CUDA computation
        try:
            print("Running simple CUDA computation test...")
            x = torch.rand(1000, 1000, device='cuda')
            y = torch.matmul(x, x.t())
            print(f"CUDA computation successful, result shape: {y.shape}")
        except Exception as e:
            print(f"CUDA computation failed: {e}")
    else:
        print("CUDA is not available.")
        print("Check that you have a compatible GPU and CUDA installation.")

if __name__ == "__main__":
    test_cuda()
```

Run with:
```bash
python cuda_test.py
```

### Debug Mode for Detailed CUDA Information

Set environment variables to enable detailed CUDA debug logging:

**Windows:**
```
set CUDA_LAUNCH_BLOCKING=1
set WHISPER_BATCH_LOG_LEVEL=DEBUG
whisper-batch /path/to/videos /path/to/output
```

**Linux/macOS:**
```bash
CUDA_LAUNCH_BLOCKING=1 WHISPER_BATCH_LOG_LEVEL=DEBUG whisper-batch /path/to/videos /path/to/output
```

### GPU-Z Analysis

1. Download [GPU-Z](https://www.techpowerup.com/gpuz/)
2. Run GPU-Z while running WhisperBatch
3. Check for:
   - GPU Load (should increase during transcription)
   - Memory Used (should increase when model loads)
   - Power Consumption (should increase during processing)

If GPU values don't change during transcription, CUDA is not being properly utilized.

## Common CUDA Error Messages and Meanings

| Error Message | Meaning | Solution |
|---------------|---------|----------|
| `cudnn_ops64_9.dll not found` | cuDNN DLL is missing | Install cuDNN and copy DLLs to CUDA bin directory |
| `CUDA out of memory` | Not enough VRAM | Use smaller model or reduce batch size |
| `CUDNN_STATUS_NOT_INITIALIZED` | cuDNN initialization failed | Ensure compatible CUDA and cuDNN versions |
| `CUDA driver version is insufficient` | Driver too old | Update NVIDIA drivers |
| `no kernel image is available for execution` | GPU architecture mismatch | Use a compatible CUDA version for your GPU |
| `CUBLAS_STATUS_EXECUTION_FAILED` | cuBLAS computation error | Check CUDA and driver compatibility |
| `device-side assert triggered` | CUDA kernel error | Update drivers or use CPU mode |

## When to Use CPU Instead of GPU

Sometimes, using CPU is better than troubleshooting CUDA issues:

1. **For Short Files**: For files under 1 minute, the model loading overhead might make GPU processing slower
2. **For Basic Models**: The tiny.en and base.en models run reasonably fast on modern CPUs
3. **For Reliability**: If you need 100% reliability without CUDA issues
4. **For Development**: When developing or testing new features

To force CPU mode:
```bash
whisper-batch /path/to/videos /path/to/output --device cpu
``` 