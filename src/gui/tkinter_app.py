import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
import threading
import queue
import logging
from pathlib import Path

# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from whisper_batch import DEFAULT_LOG_FORMAT
    import whisper_batch.file_handler as file_handler
    from whisper_batch.main import process_videos
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure you are running from the project root or the script can find the 'src' directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format=DEFAULT_LOG_FORMAT)
logger = logging.getLogger("gui")

# List of available Whisper models 
AVAILABLE_MODELS = [
    "tiny", "tiny.en", 
    "base", "base.en", 
    "small", "small.en",
    "medium", "medium.en",
    "large", "large-v3"
]

# List of languages (partial list - would need to be expanded)
LANGUAGES = [
    ("Auto Detect", None),
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Chinese", "zh"),
    ("Russian", "ru"),
    ("Portuguese", "pt"),
    ("Arabic", "ar"),
    # Add more languages as needed
]

class LoggingHandler(logging.Handler):
    """Custom logging handler that redirects logs to a tkinter Text widget."""
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        msg = self.format(record)
        
        # Tkinter is not thread-safe, so we need to use after() 
        # to schedule updates from non-main threads
        self.text_widget.after(0, self._append_log, msg)
    
    def _append_log(self, msg):
        """Append log message to text widget"""
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, msg + '\n')
        self.text_widget.see(tk.END)  # Scroll to the end
        self.text_widget.configure(state='disabled')

class WhisperBatchGUI(tk.Tk):
    """Main Tkinter GUI Application for Whisper Batch."""
    
    def __init__(self):
        super().__init__()
        
        self.title("Whisper Batch - Video Transcription Tool")
        self.geometry("800x600")
        self.minsize(640, 480)
        
        # Variables for form fields
        self.input_dir = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model_var = tk.StringVar(value="base.en")  # Default model
        self.language_var = tk.StringVar(value="Auto Detect")
        self.output_format_var = tk.StringVar(value="txt")
        self.use_gpu_var = tk.BooleanVar(value=True)
        self.skip_processed_var = tk.BooleanVar(value=True)
        self.device_var = tk.StringVar(value="auto")
        
        # Processing status variables
        self.is_processing = False
        self.process_thread = None
        self.progress_queue = queue.Queue()
        
        # Set up UI components
        self._create_widgets()
        self._setup_layout()
        
        # Configure logging to the GUI
        self._setup_logging()
        
        # Display system info in the log
        self._log_system_info()
        
        # Add a protocol handler for the close event
        self.protocol("WM_DELETE_WINDOW", self._on_close)
    
    def _create_widgets(self):
        """Create all the widgets for the GUI."""
        # --- Frame for directory selection ---
        self.dir_frame = ttk.LabelFrame(self, text="Directories")
        
        # Input directory selection
        ttk.Label(self.dir_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.dir_frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.dir_frame, text="Browse...", command=self._browse_input_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # Output directory selection
        ttk.Label(self.dir_frame, text="Output Directory:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        ttk.Entry(self.dir_frame, textvariable=self.output_dir, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.dir_frame, text="Browse...", command=self._browse_output_dir).grid(row=1, column=2, padx=5, pady=5)
        
        # --- Frame for transcription settings ---
        self.settings_frame = ttk.LabelFrame(self, text="Transcription Settings")
        
        # Model selection
        ttk.Label(self.settings_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        model_combo = ttk.Combobox(self.settings_frame, textvariable=self.model_var, values=AVAILABLE_MODELS, state="readonly")
        model_combo.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Language selection
        ttk.Label(self.settings_frame, text="Language:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        language_combo = ttk.Combobox(self.settings_frame, textvariable=self.language_var, 
                                      values=[lang[0] for lang in LANGUAGES], state="readonly")
        language_combo.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Device selection
        ttk.Label(self.settings_frame, text="Device:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        device_combo = ttk.Combobox(self.settings_frame, textvariable=self.device_var, values=["auto", "cuda", "cpu"], state="readonly")
        device_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Output format selection
        ttk.Label(self.settings_frame, text="Output Format:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=5)
        format_frame = ttk.Frame(self.settings_frame)
        format_frame.grid(row=3, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Radiobutton(format_frame, text="Text (.txt)", variable=self.output_format_var, value="txt").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(format_frame, text="Subtitles (.srt)", variable=self.output_format_var, value="srt").pack(side=tk.LEFT, padx=5)
        
        # Use GPU checkbox
        ttk.Checkbutton(self.settings_frame, text="Use GPU (if available)", variable=self.use_gpu_var).grid(
            row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # Skip already processed files
        ttk.Checkbutton(self.settings_frame, text="Skip already processed files", variable=self.skip_processed_var).grid(
            row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        
        # --- Frame for controls and progress ---
        self.control_frame = ttk.Frame(self)
        
        # Start/Cancel button
        self.start_button = ttk.Button(self.control_frame, text="Start Transcription", command=self._start_transcription)
        self.start_button.pack(side=tk.LEFT, padx=5, pady=10)
        
        self.cancel_button = ttk.Button(self.control_frame, text="Cancel", command=self._cancel_transcription, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=5, pady=10)
        
        # Progress indicator
        self.progress_frame = ttk.LabelFrame(self, text="Progress")
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        self.status_label = ttk.Label(self.progress_frame, text="Ready")
        self.status_label.pack(anchor=tk.W, padx=5)
        
        # Log output
        self.log_frame = ttk.LabelFrame(self, text="Log")
        
        self.log_text = tk.Text(self.log_frame, height=10, wrap=tk.WORD, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        log_scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _setup_layout(self):
        """Arrange the widgets in the layout."""
        # Configure the layout to be responsive
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)  # Make the log expand
        
        # Place the frames in the main window
        self.dir_frame.grid(row=0, column=0, padx=10, pady=10, sticky=tk.EW)
        self.settings_frame.grid(row=1, column=0, padx=10, pady=10, sticky=tk.EW)
        self.control_frame.grid(row=2, column=0, padx=10, pady=10, sticky=tk.EW)
        self.progress_frame.grid(row=3, column=0, padx=10, pady=10, sticky=tk.EW)
        self.log_frame.grid(row=4, column=0, padx=10, pady=10, sticky=tk.NSEW)
    
    def _setup_logging(self):
        """Configure logging to output to the GUI."""
        log_handler = LoggingHandler(self.log_text)
        log_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))
        
        # Add the handler to the root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(log_handler)
        
        logger.info("GUI application started")
    
    def _log_system_info(self):
        """Log system information for debugging purposes."""
        try:
            import sys
            import platform
            import torch
            
            logger.info("--- System Information ---")
            logger.info(f"OS: {platform.system()} {platform.version()}")
            logger.info(f"Python: {sys.version}")
            
            # Check for CUDA and GPU
            cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {cuda_available}")
            
            if cuda_available:
                try:
                    device_count = torch.cuda.device_count()
                    logger.info(f"CUDA device count: {device_count}")
                    
                    for i in range(device_count):
                        device_name = torch.cuda.get_device_name(i)
                        device_capability = torch.cuda.get_device_capability(i)
                        logger.info(f"CUDA device {i}: {device_name} (Compute Capability: {device_capability})")
                except Exception as e:
                    logger.warning(f"Error getting CUDA device info: {e}")
            
            # Check for ffmpeg
            import shutil
            ffmpeg_path = shutil.which("ffmpeg")
            logger.info(f"FFmpeg found: {ffmpeg_path is not None}")
            if ffmpeg_path:
                logger.info(f"FFmpeg path: {ffmpeg_path}")
                
            logger.info("-------------------------")
        except Exception as e:
            logger.error(f"Error logging system info: {e}")
    
    def _browse_input_dir(self):
        """Open a file dialog to select the input directory."""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:  # if a directory is selected
            self.input_dir.set(directory)
            # If output directory is empty, set a default (input_dir/transcripts)
            if not self.output_dir.get():
                self.output_dir.set(os.path.join(directory, "transcripts"))
    
    def _browse_output_dir(self):
        """Open a file dialog to select the output directory."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:  # if a directory is selected
            self.output_dir.set(directory)
    
    def _start_transcription(self):
        """Start the transcription process."""
        # Validate inputs
        if not self._validate_inputs():
            return
        
        # Get selected language code from the language name
        selected_language = self.language_var.get()
        language_code = None
        for lang_name, lang_code in LANGUAGES:
            if lang_name == selected_language:
                language_code = lang_code
                break
        
        # Prepare the parameters for the processing function
        input_dir = Path(self.input_dir.get())
        output_dir = Path(self.output_dir.get())
        model_size = self.model_var.get()
        device = self.device_var.get()
        output_format = self.output_format_var.get()
        skip_processed = self.skip_processed_var.get()
        
        # Update UI state
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.status_label.config(text="Processing...")
        
        # Start processing in a separate thread
        self.process_thread = threading.Thread(
            target=self._run_processing,
            args=(input_dir, output_dir, model_size, language_code, device, output_format, skip_processed),
            daemon=True
        )
        self.process_thread.start()
        
        # Start monitoring the progress
        self.after(100, self._check_progress)
    
    def _run_processing(self, input_dir, output_dir, model_size, language, device, output_format, skip_processed):
        """Run the video processing (to be called in a separate thread)."""
        try:
            # Configure progress reporting
            def progress_callback(current, total, filename=""):
                self.progress_queue.put((current, total, filename))
            
            # Count files to process
            video_files = file_handler.find_video_files(input_dir)
            if not video_files:
                self.progress_queue.put(("error", "No video files found in the selected directory."))
                return
            
            # Before processing, check for potential issues
            if device == "cuda":
                # Log CUDA version and device information
                import torch
                
                if not torch.cuda.is_available():
                    logger.warning("CUDA not available despite being requested. Will use CPU instead.")
                    device = "cpu"
                else:
                    try:
                        # Test CUDA initialization
                        device_name = torch.cuda.get_device_name(0)
                        logger.info(f"Using CUDA device: {device_name}")
                    except Exception as e:
                        logger.warning(f"CUDA initialization error: {e}")
                        logger.warning("Falling back to CPU. If you need GPU acceleration, make sure CUDA is properly installed.")
                        device = "cpu"
            
            # Log FFmpeg version
            import shutil
            import subprocess
            ffmpeg_path = shutil.which("ffmpeg")
            if not ffmpeg_path:
                error_msg = "FFmpeg not found in PATH. Cannot process videos."
                logger.error(error_msg)
                self.progress_queue.put(("error", error_msg))
                return
            else:
                try:
                    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
                    ffmpeg_version = result.stdout.split('\n')[0]
                    logger.info(f"Using {ffmpeg_version}")
                except Exception as e:
                    logger.warning(f"Error checking FFmpeg version: {e}")
            
            # Call the processing function
            try:
                process_videos(
                    input_path=input_dir,
                    output_path=output_dir,
                    model_size=model_size,
                    language=language,
                    device=device,
                    compute_type="default",
                    output_format=output_format,
                    skip_processed=skip_processed,
                    progress_callback=progress_callback
                )
                
                # Signal completion
                self.progress_queue.put(("complete", None))
            except Exception as e:
                logger.error(f"Error during video processing: {e}", exc_info=True)
                error_message = str(e)
                
                # Check for specific error types and provide more helpful messages
                error_lower = error_message.lower()
                if "cuda" in error_lower or "gpu" in error_lower or "cudnn" in error_lower:
                    error_message = f"GPU error: {error_message}\n\nTry running with CPU mode enabled (uncheck 'Use GPU')."
                elif "ffmpeg" in error_lower:
                    error_message = f"FFmpeg error: {error_message}\n\nMake sure FFmpeg is properly installed and in your PATH."
                elif "permission" in error_lower or "access" in error_lower:
                    error_message = f"File access error: {error_message}\n\nCheck if you have permissions to access the input/output directories."
                
                self.progress_queue.put(("error", error_message))
            
        except Exception as e:
            # Signal error
            logger.error(f"Error during processing setup: {e}", exc_info=True)
            self.progress_queue.put(("error", str(e)))
    
    def _show_error_dialog(self, title, message, details=None):
        """Show an enhanced error dialog with optional details."""
        error_dialog = tk.Toplevel(self)
        error_dialog.title(title)
        error_dialog.geometry("500x400")
        error_dialog.minsize(400, 300)
        error_dialog.transient(self)  # Set to be on top of the main window
        error_dialog.grab_set()  # Modal
        
        # Make dialog resizable
        error_dialog.columnconfigure(0, weight=1)
        error_dialog.rowconfigure(1, weight=1)
        
        # Error icon and message at the top
        header_frame = ttk.Frame(error_dialog)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Use standard error icon (on Windows)
        if hasattr(tk, 'PhotoImage'):
            try:
                error_icon = tk.PhotoImage(file="")  # This trick loads the system error icon on Windows
                ttk.Label(header_frame, image=error_icon).pack(side=tk.LEFT, padx=10)
                error_dialog.error_icon = error_icon  # Keep a reference
            except:
                pass
        
        ttk.Label(header_frame, text=message, wraplength=400).pack(side=tk.LEFT, padx=5)
        
        # Details in scrolled text if provided
        if details:
            details_frame = ttk.LabelFrame(error_dialog, text="Details")
            details_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
            
            details_frame.columnconfigure(0, weight=1)
            details_frame.rowconfigure(0, weight=1)
            
            details_text = tk.Text(details_frame, wrap=tk.WORD, width=60, height=10)
            details_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            
            scrollbar = ttk.Scrollbar(details_frame, command=details_text.yview)
            scrollbar.grid(row=0, column=1, sticky="ns", pady=5)
            details_text.configure(yscrollcommand=scrollbar.set)
            
            details_text.insert(tk.END, details)
            details_text.configure(state="disabled")
            
            # Add a "Copy to Clipboard" button
            def copy_to_clipboard():
                self.clipboard_clear()
                self.clipboard_append(details)
                self.update()  # Required for clipboard to work
            
            ttk.Button(details_frame, text="Copy to Clipboard", command=copy_to_clipboard).grid(
                row=1, column=0, columnspan=2, pady=5)
        
        # Buttons at the bottom
        button_frame = ttk.Frame(error_dialog)
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        
        # View log button (shows log file location)
        def view_log():
            import os
            from pathlib import Path
            # Get log directory from setup_logging
            logs_dir = Path.home() / 'whisper_batch_logs'
            if logs_dir.exists():
                if hasattr(os, 'startfile'):  # Windows
                    os.startfile(logs_dir)
                elif os.name == 'posix':  # macOS and Linux
                    import subprocess
                    try:
                        subprocess.run(['xdg-open', logs_dir])  # Linux
                    except FileNotFoundError:
                        try:
                            subprocess.run(['open', logs_dir])  # macOS
                        except:
                            pass
        
        ttk.Button(button_frame, text="View Logs", command=view_log).pack(side=tk.LEFT, padx=5)
        
        # OK button (closes the dialog)
        ttk.Button(button_frame, text="OK", command=error_dialog.destroy).pack(side=tk.RIGHT, padx=5)
        
        # Center the dialog on the main window
        error_dialog.update_idletasks()
        width = error_dialog.winfo_width()
        height = error_dialog.winfo_height()
        x = self.winfo_rootx() + (self.winfo_width() - width) // 2
        y = self.winfo_rooty() + (self.winfo_height() - height) // 2
        error_dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Wait for the dialog to be closed
        error_dialog.wait_window()

    def _check_progress(self):
        """Check the progress queue and update the UI accordingly."""
        try:
            while not self.progress_queue.empty():
                msg = self.progress_queue.get_nowait()
                
                if msg[0] == "complete":
                    # Processing completed
                    self.progress_bar['value'] = 100
                    self.status_label.config(text="Transcription complete")
                    self._reset_processing_state()
                    messagebox.showinfo("Complete", "Transcription process has completed successfully!")
                    return
                
                elif msg[0] == "error":
                    # Error occurred
                    error_message = msg[1]
                    self.status_label.config(text=f"Error: {error_message[:50]}..." if len(error_message) > 50 else f"Error: {error_message}")
                    self._reset_processing_state()
                    
                    # Get the last few lines from the log for details
                    import traceback
                    details = f"Error: {error_message}\n\n"
                    details += "Please check the log file for more details.\n"
                    
                    # Show the enhanced error dialog
                    self._show_error_dialog(
                        title="Transcription Error",
                        message="An error occurred during the transcription process.",
                        details=details
                    )
                    return
                
                else:
                    # Update progress
                    current, total, filename = msg
                    progress_pct = (current / total) * 100 if total > 0 else 0
                    self.progress_bar['value'] = progress_pct
                    self.status_label.config(text=f"Processing: {filename} ({current}/{total})")
        
        except Exception as e:
            logger.error(f"Error checking progress: {e}", exc_info=True)
        
        # Schedule the next check if still processing
        if self.is_processing:
            self.after(100, self._check_progress)
    
    def _cancel_transcription(self):
        """Cancel the transcription process."""
        if self.is_processing:
            # Set a flag or signal to stop the processing
            # (Would need to modify process_videos to check for cancellation)
            logger.info("Cancelling transcription...")
            self.status_label.config(text="Cancelling...")
            
            # For now, just reset the state 
            # In a real implementation, you'd need proper cancellation support
            self._reset_processing_state()
    
    def _reset_processing_state(self):
        """Reset the UI state after processing is done or cancelled."""
        self.is_processing = False
        self.start_button.config(state=tk.NORMAL)
        self.cancel_button.config(state=tk.DISABLED)
    
    def _validate_inputs(self):
        """Validate the user inputs before starting processing."""
        # Check if input directory exists
        if not self.input_dir.get():
            messagebox.showerror("Error", "Please select an input directory.")
            return False
        
        if not os.path.isdir(self.input_dir.get()):
            messagebox.showerror("Error", "The specified input directory does not exist.")
            return False
        
        # Check if output directory is specified
        if not self.output_dir.get():
            messagebox.showerror("Error", "Please specify an output directory.")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir.get(), exist_ok=True)
        
        return True
    
    def _on_close(self):
        """Handle the close event."""
        if self.is_processing:
            if messagebox.askyesno("Confirm Exit", "A transcription is currently running. Are you sure you want to exit?"):
                self._cancel_transcription()
                self.destroy()
        else:
            self.destroy()

def main():
    """Start the Whisper Batch GUI application."""
    app = WhisperBatchGUI()
    app.mainloop()

if __name__ == "__main__":
    main() 