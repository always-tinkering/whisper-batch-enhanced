# **Automated Video Transcription Tool Product Document**  
**Version 0.1**  
**Date:** April 9, 2025  

---

## **Objective**  
Develop a cross-platform (Windows/macOS/Linux) Python GUI application that automates the extraction of audio transcripts from video files stored in nested directories. The tool will process all supported video files in a user-selected folder and its subfolders, transcribe them using a local speech-to-text engine (e.g., Whisper.cpp), and save the text outputs in a structured directory mirroring the input folder.  

---

## **Key Features**  
### **1. Folder-Based Batch Processing**  
- Recursively scan user-selected directories for video files (e.g., MP4, MKV, AVI).  
- Support for common video formats via FFmpeg integration.  
- Skip already processed files to avoid redundancy.  

### **2. Transcription Engine**  
- Integrate **Whisper.cpp** or **faster-whisper** for offline, high-accuracy transcription.  
- Allow users to select model size (e.g., `tiny.en`, `base.en`, `large-v3`), balancing speed and accuracy.  
- GPU acceleration support (CUDA for Nvidia, Metal for macOS) for faster processing.  

### **3. Output Management**  
- Create an output directory mirroring the input folder structure.  
- Save transcripts as `.txt` files with the same base name as the input video.  
- Optional formats: `.srt` (subtitles), `.vtt`.  

### **4. GUI Interface**  
- **Input/Output Selection:**  
  - Folder picker for input directory.  
  - Default output directory (e.g., `input_folder/transcripts`) with customization.  
- **Progress Tracking:**  
  - Real-time progress bar and file counter.  
  - Error logging for failed files (e.g., corrupted videos).  
- **Configuration Panel:**  
  - Model selection dropdown.  
  - Language selection (default: auto-detect).  
  - Toggle for GPU acceleration.  

### **5. Additional Requirements**  
- **Low Resource Mode:** Limit parallel processes to avoid overloading systems with limited RAM/CPU.  
- **File Naming Conventions:** Handle special characters and spaces in filenames.  
- **Privacy:** No data upload—process entirely locally.  

---

## **Technical Specifications**  
### **Dependencies**  
- Python 3.9+  
- **Backend:**  
  - `whisper-cpp` or `faster-whisper`  
  - `ffmpeg-python` (audio extraction)  
  - `pydub` (audio format conversion)  
- **GUI Framework:**  
  - `PyQt6` or `Tkinter` (lightweight option)  
- **Utilities:**  
  - `tqdm` (progress bars)  
  - `watchdog` (directory monitoring for future features)  

### **Workflow**  
1. **User selects input folder** and output directory.  
2. **Scan directory** recursively for supported video files.  
3. **Extract audio** from each video using FFmpeg:  
   ```python  
   ffmpeg.input(video_path).output(audio_path, ac=1, ar=16000).run()  
   ```
4. **Transcribe audio** with Whisper:  
   ```python  
   model = whisper.load_model("base.en")  
   result = model.transcribe(audio_path, beam_size=5)  
   ```
5. **Save transcript** to output directory, preserving folder structure.  

---

## **User Interface Mockup**  
GUI Layout  
- **A:** Input folder selection button.  
- **B:** Output folder selection + path display.  
- **C:** Model/language settings dropdowns.  
- **D:** Start/Cancel/Pause buttons.  
- **E:** Progress bar + status log.  

---

## **Error Handling**  
- **Invalid Files:** Skip unsupported formats, log errors to `errors.txt`.  
- **CUDA/GPU Failures:** Fall back to CPU mode automatically.  
- **Permissions:** Alert users if output directories are unwritable.  

---

## **Development Roadmap**  
### **Phase 1: MVP (4 Weeks)**  
- Core transcription pipeline with CLI support.  
- Basic Tkinter GUI for folder selection.  
- Testing on Windows (PyInstaller executable).  

### **Phase 2: Enhanced GUI (2 Weeks)**  
- PyQt6 interface with progress tracking.  
- macOS/Linux packaging (`.app`, `.deb`, `.rpm`).  

### **Phase 3: Advanced Features (2 Weeks)**  
- Parallel processing for multi-core systems.  
- Subtitles export (SRT/VTT).  

---

## **References**  
1. Reddit post on bulk Whisper transcriptions.  
2. Batch execution in faster-whisper.  
3. Python transcription tutorial using SpeechRecognition.  
4. Local-first design inspired by Whisper Batch Transcriber.  
5. Cross-platform packaging from transcribe-anything.  

--- 

**Attachments:**  
- Sample project structure (GitHub repo).  
- Whisper model comparison chart.  
- FFmpeg installation guide for Windows/macOS/Linux.  

This document provides a blueprint for developing a user-friendly, scalable solution to automate video transcription workflows. The tool prioritizes privacy, flexibility, and ease of use while leveraging state-of-the-art speech recognition models.