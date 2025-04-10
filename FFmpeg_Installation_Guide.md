# FFmpeg Installation Guide for Windows

Follow these steps to install FFmpeg on your Windows system:

## Option 1: Direct Download and Manual Setup

1. **Download FFmpeg**:
   - Visit [FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases)
   - Download the latest `ffmpeg-master-latest-win64-gpl.zip` file

2. **Extract and Set Up**:
   - Create a folder at `C:\ffmpeg`
   - Extract the downloaded zip file contents to this folder
   - Inside the extracted folder, find the `bin` directory which contains `ffmpeg.exe`

3. **Add to System PATH**:
   - Right-click on "This PC" or "My Computer" and select "Properties"
   - Click on "Advanced system settings"
   - Click the "Environment Variables" button
   - Under "System variables", find the "Path" variable, select it and click "Edit"
   - Click "New" and add `C:\ffmpeg\bin` (or the full path to wherever you extracted the bin folder)
   - Click "OK" on all dialogs to save the changes

4. **Verify Installation**:
   - Open a new Command Prompt or PowerShell window
   - Type `ffmpeg -version` and press Enter
   - You should see version information if FFmpeg was installed correctly

## Option 2: Using PowerShell (Admin required)

Run PowerShell as Administrator and execute:

```powershell
# Create directory
New-Item -ItemType Directory -Force -Path C:\ffmpeg

# Download FFmpeg
$url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
$output = "C:\ffmpeg\ffmpeg.zip"
Invoke-WebRequest -Uri $url -OutFile $output

# Extract the zip file
Expand-Archive -LiteralPath $output -DestinationPath C:\ffmpeg -Force
Remove-Item $output  # Clean up the zip file

# Find the bin directory and add to PATH
$binDir = Get-ChildItem -Path C:\ffmpeg -Recurse -Filter "bin" | Where-Object { $_.PSIsContainer } | Select-Object -First 1 -ExpandProperty FullName
[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'Machine') + ";$binDir", 'Machine')

# Verify
& "$binDir\ffmpeg.exe" -version
```

## After Installation

After installing FFmpeg, you'll need to:

1. Close and reopen any Command Prompt or PowerShell windows for the PATH changes to take effect
2. Run the Whisper Batch application again:
   ```
   python src/whisper_batch/main.py test/videos --log_level DEBUG
   ``` 