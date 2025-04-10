"""
WhisperBatch Qt GUI Application

Modern Windows 11 style GUI for WhisperBatch using QtPy with PyQt6 backend.
"""

import sys
import os
import threading
import queue
import logging
from pathlib import Path

# Ensure the src directory is in the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from qtpy import QtWidgets, QtCore, QtGui
    from whisper_batch import DEFAULT_LOG_FORMAT
    import whisper_batch.file_handler as file_handler
    from whisper_batch.main import process_videos
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure you are running from the project root and have installed all requirements.")
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

# List of languages
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

class LogTextEdit(QtWidgets.QPlainTextEdit):
    """Custom text edit widget for logging."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)  # Limit number of lines for performance
        
        # Custom font
        font = QtGui.QFont("Segoe UI", 9)
        self.setFont(font)

class LoggingHandler(logging.Handler):
    """Custom logging handler that redirects logs to a Qt text widget."""
    
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        
    def emit(self, record):
        msg = self.format(record)
        QtCore.QMetaObject.invokeMethod(
            self.text_widget,
            "appendPlainText",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(str, msg)
        )

class WhisperBatchQt(QtWidgets.QMainWindow):
    """Main Qt GUI Application for Whisper Batch."""
    
    progress_update = QtCore.Signal(int, int, str)
    processing_complete = QtCore.Signal()
    processing_error = QtCore.Signal(str, str)
    # Add new signals for detailed progress
    model_loading_progress = QtCore.Signal(int)  # 0-100 percentage
    file_detection_progress = QtCore.Signal(int)  # 0-100 percentage
    current_file_progress = QtCore.Signal(int)  # 0-100 percentage
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Whisper Batch - Video Transcription Tool")
        # Increase default size to better fit content
        self.resize(1000, 800)
        self.setMinimumSize(800, 600)
        
        # Set application icon
        self._create_app_icon()
        
        # Processing status variables
        self.is_processing = False
        self.process_thread = None
        self.cancel_requested = False
        
        # Create central widget with scroll area for better handling of small window sizes
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setCentralWidget(self.scroll_area)
        
        # Create content widget to hold all components
        self.central_widget = QtWidgets.QWidget()
        self.scroll_area.setWidget(self.central_widget)
        
        # Main layout with improved spacing
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)  # Increased spacing between main sections
        
        # Setup UI components
        self._create_ui()
        
        # Connect signals
        self.progress_update.connect(self._update_progress)
        self.processing_complete.connect(self._on_processing_complete)
        self.processing_error.connect(self._show_error_dialog)
        # Connect new progress signals
        self.model_loading_progress.connect(self._update_model_loading_progress)
        self.file_detection_progress.connect(self._update_file_detection_progress)
        self.current_file_progress.connect(self._update_current_file_progress)
        
        # Configure logging to the GUI
        self._setup_logging()
        
        # Display system info in the log
        self._log_system_info()
        
        # Apply Windows 11 style
        self._apply_win11_style()
        
        # Apply shadow effects to UI elements
        self._apply_shadow_effects()
        
        # Setup resize event monitoring for overlap detection
        self._setup_overlap_detection()
    
    def _create_ui(self):
        """Create all UI elements."""
        # --- Directory selection frame ---
        dir_group = QtWidgets.QGroupBox("Directories")
        dir_layout = QtWidgets.QGridLayout(dir_group)
        dir_layout.setContentsMargins(15, 25, 15, 15)
        dir_layout.setSpacing(12)
        dir_layout.setVerticalSpacing(15)
        
        # Input directory
        input_label = QtWidgets.QLabel("Input Directory:")
        input_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        input_label.setStyleSheet("font-weight: bold;")
        dir_layout.addWidget(input_label, 0, 0)
        
        self.input_dir_edit = QtWidgets.QLineEdit()
        dir_layout.addWidget(self.input_dir_edit, 0, 1)
        
        input_browse_btn = QtWidgets.QPushButton("Browse...")
        input_browse_btn.clicked.connect(self._browse_input_dir)
        input_browse_btn.setFixedWidth(100)
        dir_layout.addWidget(input_browse_btn, 0, 2)
        
        # Output directory
        output_label = QtWidgets.QLabel("Output Directory:")
        output_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        output_label.setStyleSheet("font-weight: bold;")
        dir_layout.addWidget(output_label, 1, 0)
        
        self.output_dir_edit = QtWidgets.QLineEdit()
        dir_layout.addWidget(self.output_dir_edit, 1, 1)
        
        output_browse_btn = QtWidgets.QPushButton("Browse...")
        output_browse_btn.clicked.connect(self._browse_output_dir)
        output_browse_btn.setFixedWidth(100)
        dir_layout.addWidget(output_browse_btn, 1, 2)
        
        # Set column stretch to make the text field expand
        dir_layout.setColumnStretch(1, 1)
        
        self.main_layout.addWidget(dir_group)
        
        # --- Transcription settings frame ---
        settings_group = QtWidgets.QGroupBox("Transcription Settings")
        
        # Create grid layout for settings with better alignment
        settings_layout = QtWidgets.QGridLayout(settings_group)
        settings_layout.setContentsMargins(15, 25, 15, 15)
        settings_layout.setVerticalSpacing(15)
        settings_layout.setHorizontalSpacing(15)
        
        # Create a horizontal layout for the model, language, and device fields
        field_container = QtWidgets.QWidget()
        field_layout = QtWidgets.QHBoxLayout(field_container)
        field_layout.setContentsMargins(0, 0, 0, 0)
        field_layout.setSpacing(20)
        
        # Model selection
        model_widget = QtWidgets.QWidget()
        model_layout = QtWidgets.QHBoxLayout(model_widget)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(10)
        
        model_label = QtWidgets.QLabel("Model:")
        model_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        model_label.setFixedWidth(60)
        model_label.setStyleSheet("font-weight: bold;")
        model_layout.addWidget(model_label)
        
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(AVAILABLE_MODELS)
        self.model_combo.setCurrentText("base.en")
        self.model_combo.setMinimumWidth(120)
        model_layout.addWidget(self.model_combo)
        
        field_layout.addWidget(model_widget)
        
        # Language selection
        language_widget = QtWidgets.QWidget()
        language_layout = QtWidgets.QHBoxLayout(language_widget)
        language_layout.setContentsMargins(0, 0, 0, 0)
        language_layout.setSpacing(10)
        
        language_label = QtWidgets.QLabel("Language:")
        language_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        language_label.setFixedWidth(70)
        language_label.setStyleSheet("font-weight: bold;")
        language_layout.addWidget(language_label)
        
        self.language_combo = QtWidgets.QComboBox()
        for lang_name, _ in LANGUAGES:
            self.language_combo.addItem(lang_name)
        self.language_combo.setMinimumWidth(120)
        language_layout.addWidget(self.language_combo)
        
        field_layout.addWidget(language_widget)
        
        # Device selection
        device_widget = QtWidgets.QWidget()
        device_layout = QtWidgets.QHBoxLayout(device_widget)
        device_layout.setContentsMargins(0, 0, 0, 0)
        device_layout.setSpacing(10)
        
        device_label = QtWidgets.QLabel("Device:")
        device_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        device_label.setFixedWidth(60)
        device_label.setStyleSheet("font-weight: bold;")
        device_layout.addWidget(device_label)
        
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cpu"])
        self.device_combo.setMinimumWidth(80)
        device_layout.addWidget(self.device_combo)
        
        field_layout.addWidget(device_widget)
        field_layout.addStretch(1)
        
        # Add the horizontal field container to the main layout
        settings_layout.addWidget(field_container, 0, 0, 1, 2)
        
        # Output format
        format_label = QtWidgets.QLabel("Output Format:")
        format_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        format_label.setStyleSheet("font-weight: bold;")
        settings_layout.addWidget(format_label, 1, 0)
        
        format_container = QtWidgets.QWidget()
        format_layout = QtWidgets.QHBoxLayout(format_container)
        format_layout.setContentsMargins(0, 0, 0, 0)
        format_layout.setSpacing(20)
        format_layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        
        button_group = QtWidgets.QButtonGroup(format_container)
        self.format_txt_radio = QtWidgets.QRadioButton("Text (.txt)")
        self.format_txt_radio.setChecked(True)
        self.format_srt_radio = QtWidgets.QRadioButton("Subtitles (.srt)")
        
        button_group.addButton(self.format_txt_radio)
        button_group.addButton(self.format_srt_radio)
        
        format_layout.addWidget(self.format_txt_radio)
        format_layout.addWidget(self.format_srt_radio)
        format_layout.addStretch()
        
        settings_layout.addWidget(format_container, 1, 1)
        
        # Extra spacing
        spacer = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        settings_layout.addItem(spacer, 2, 0)
        
        # Use GPU checkbox (with proper indentation)
        self.use_gpu_check = QtWidgets.QCheckBox("Use GPU (if available)")
        self.use_gpu_check.setChecked(True)
        self.use_gpu_check.setStyleSheet("margin-left: 40px; font-weight: bold;")
        settings_layout.addWidget(self.use_gpu_check, 3, 0, 1, 2)
        
        # Skip processed files checkbox
        self.skip_processed_check = QtWidgets.QCheckBox("Skip already processed files")
        self.skip_processed_check.setChecked(True)
        self.skip_processed_check.setStyleSheet("margin-left: 40px; font-weight: bold;")
        settings_layout.addWidget(self.skip_processed_check, 4, 0, 1, 2)
        
        # Set column stretch
        settings_layout.setColumnStretch(1, 1)
        
        self.main_layout.addWidget(settings_group)
        
        # --- Controls frame ---
        control_frame = QtWidgets.QFrame()
        control_layout = QtWidgets.QHBoxLayout(control_frame)
        control_layout.setContentsMargins(5, 10, 5, 5)
        
        self.start_button = QtWidgets.QPushButton("Start Transcription")
        self.start_button.clicked.connect(self._start_transcription)
        self.start_button.setMinimumHeight(40)
        self.start_button.setMinimumWidth(150)
        self.start_button.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_transcription)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setMinimumHeight(40)
        self.cancel_button.setMinimumWidth(100)
        
        control_layout.addWidget(self.start_button)
        control_layout.addSpacing(10)
        control_layout.addWidget(self.cancel_button)
        control_layout.addStretch()
        
        self.main_layout.addWidget(control_frame)
        
        # --- Progress frame ---
        progress_group = QtWidgets.QGroupBox("Progress")
        progress_layout = QtWidgets.QVBoxLayout(progress_group)
        progress_layout.setContentsMargins(15, 25, 15, 15)
        progress_layout.setSpacing(10)
        
        # Add detailed progress wheels
        detailed_progress_frame = QtWidgets.QFrame()
        detailed_progress_layout = QtWidgets.QHBoxLayout(detailed_progress_frame)
        detailed_progress_layout.setContentsMargins(0, 0, 0, 0)
        detailed_progress_layout.setSpacing(15)
        
        # Current file progress wheel
        self.current_file_progress = QtWidgets.QProgressBar()
        self.current_file_progress.setRange(0, 100)
        self.current_file_progress.setValue(0)
        self.current_file_progress.setMaximumWidth(100)
        self.current_file_progress.setTextVisible(True)
        self.current_file_progress.setFormat("File: %p%")
        # Set circular style for progress bars
        self.current_file_progress.setStyleSheet("""
            QProgressBar {
                border-radius: 50px;
                text-align: center;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 50px;
            }
        """)
        current_file_layout = QtWidgets.QVBoxLayout()
        current_file_layout.addWidget(self.current_file_progress)
        current_file_layout.addWidget(QtWidgets.QLabel("Current File"), alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        detailed_progress_layout.addLayout(current_file_layout)
        
        # Model loading progress wheel
        self.model_loading_progress = QtWidgets.QProgressBar()
        self.model_loading_progress.setRange(0, 100)
        self.model_loading_progress.setValue(0)
        self.model_loading_progress.setMaximumWidth(100)
        self.model_loading_progress.setTextVisible(True)
        self.model_loading_progress.setFormat("Model: %p%")
        self.model_loading_progress.setStyleSheet("""
            QProgressBar {
                border-radius: 50px;
                text-align: center;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 50px;
            }
        """)
        model_loading_layout = QtWidgets.QVBoxLayout()
        model_loading_layout.addWidget(self.model_loading_progress)
        model_loading_layout.addWidget(QtWidgets.QLabel("Model Loading"), alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        detailed_progress_layout.addLayout(model_loading_layout)
        
        # File detection progress wheel
        self.file_detection_progress = QtWidgets.QProgressBar()
        self.file_detection_progress.setRange(0, 100)
        self.file_detection_progress.setValue(0)
        self.file_detection_progress.setMaximumWidth(100)
        self.file_detection_progress.setTextVisible(True)
        self.file_detection_progress.setFormat("Files: %p%")
        self.file_detection_progress.setStyleSheet("""
            QProgressBar {
                border-radius: 50px;
                text-align: center;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 50px;
            }
        """)
        file_detection_layout = QtWidgets.QVBoxLayout()
        file_detection_layout.addWidget(self.file_detection_progress)
        file_detection_layout.addWidget(QtWidgets.QLabel("File Detection"), alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        detailed_progress_layout.addLayout(file_detection_layout)
        
        # Add all detailed progress widgets to the progress layout
        progress_layout.addWidget(detailed_progress_frame)
        
        # Add a separator line
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        progress_layout.addWidget(separator)
        
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(20)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Overall: %p%")
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setContentsMargins(5, 0, 0, 0)
        self.status_label.setStyleSheet("font-weight: bold;")
        progress_layout.addWidget(self.status_label)
        
        self.main_layout.addWidget(progress_group)
        
        # --- Log frame ---
        log_group = QtWidgets.QGroupBox("Log")
        log_layout = QtWidgets.QVBoxLayout(log_group)
        log_layout.setContentsMargins(15, 25, 15, 15)
        log_layout.setSpacing(10)
        
        self.log_text = LogTextEdit()
        log_layout.addWidget(self.log_text)
        
        self.main_layout.addWidget(log_group, 1)  # Give log widget stretch priority
    
    def _apply_shadow_effects(self):
        """Apply shadow effects to UI elements for a modern look."""
        # Create shadow effect for buttons
        for button in self.findChildren(QtWidgets.QPushButton):
            shadow = QtWidgets.QGraphicsDropShadowEffect()
            shadow.setBlurRadius(10)
            shadow.setColor(QtGui.QColor(0, 0, 0, 30))
            shadow.setOffset(0, 2)
            button.setGraphicsEffect(shadow)
        
        # Create subtle shadow for group boxes
        for group_box in self.findChildren(QtWidgets.QGroupBox):
            shadow = QtWidgets.QGraphicsDropShadowEffect()
            shadow.setBlurRadius(15)
            shadow.setColor(QtGui.QColor(0, 0, 0, 20))
            shadow.setOffset(0, 3)
            group_box.setGraphicsEffect(shadow)
    
    def _apply_win11_style(self):
        """Apply Windows 11 modern style to the application."""
        # Modern Windows 11 style with rounded corners and Mica-like effects
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f8f8f8;
            }
            QWidget {
                font-family: 'Segoe UI', sans-serif;
                font-size: 10pt;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 11pt;
                border: 1px solid #d0d0d0;
                border-radius: 8px;
                margin-top: 2.5ex;
                padding: 15px;
                background-color: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px;
                background-color: #ffffff;
                color: #0078d4;
            }
            QPushButton {
                background-color: #f0f0f0;
                border: 1px solid #d8d8d8;
                border-radius: 4px;
                padding: 5px 15px;
                color: #202020;
                font-weight: normal;
            }
            QPushButton:hover {
                background-color: #e8e8e8;
                border: 1px solid #c8c8c8;
            }
            QPushButton:pressed {
                background-color: #d8d8d8;
            }
            QPushButton:disabled {
                color: #a0a0a0;
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
            }
            #start_button {
                background-color: #0078d4;
                color: white;
                font-weight: bold;
                border: none;
            }
            #start_button:hover {
                background-color: #006cbe;
            }
            #start_button:pressed {
                background-color: #005ca3;
            }
            QComboBox {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 2px 8px;
                background-color: #ffffff;
                selection-background-color: #0078d4;
                min-height: 25px;
                color: #000000;
                font-size: 10pt;
            }
            QLineEdit {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 2px 8px;
                background-color: #ffffff;
                selection-background-color: #0078d4;
                min-height: 25px;
                color: #000000;
                font-size: 10pt;
            }
            QComboBox:focus, QLineEdit:focus {
                border: 1px solid #0078d4;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 0px;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }
            QComboBox::down-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSI2IiB2aWV3Qm94PSIwIDAgMTAgNiI+PHBhdGggZmlsbD0iIzQwNDA0MCIgZD0iTTAgMGw1IDUgNS01eiIvPjwvc3ZnPg==);
                width: 10px;
                height: 6px;
                margin-right: 8px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #d0d0d0;
                selection-background-color: #0078d4;
                selection-color: #ffffff;
                background-color: #ffffff;
                color: #000000;
            }
            QProgressBar {
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                text-align: center;
                background-color: #f5f5f5;
                height: 16px;
                font-size: 9pt;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
            QLabel {
                color: #202020;
                font-weight: normal;
            }
            QCheckBox {
                spacing: 8px;
                color: #202020;
                min-height: 22px;
                margin-left: 0px;
                padding-left: 0px;
            }
            QRadioButton {
                spacing: 8px;
                color: #202020;
                min-height: 22px;
                min-width: 100px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #d0d0d0;
                border-radius: 3px;
                background-color: #ffffff;
            }
            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #d0d0d0;
                border-radius: 9px;
                background-color: #ffffff;
            }
            QCheckBox::indicator:hover, QRadioButton::indicator:hover {
                border: 1px solid #0078d4;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border: 1px solid #0078d4;
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMCIgaGVpZ2h0PSI4IiB2aWV3Qm94PSIwIDAgMTAgOCI+PHBhdGggZmlsbD0iI2ZmZmZmZiIgZD0iTTggMEwzLjUgNC41IDIgM2wtMiAyIDMuNSAzLjVMMTAgMmwtMi0yeiIvPjwvc3ZnPg==);
                padding: 0px;
            }
            QRadioButton::indicator:checked {
                background-color: #ffffff;
                border: 1px solid #0078d4;
            }
            QRadioButton::indicator:checked {
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI4IiBoZWlnaHQ9IjgiIHZpZXdCb3g9IjAgMCA4IDgiPjxjaXJjbGUgZmlsbD0iIzAwNzhkNCIgY3g9IjQiIGN5PSI0IiByPSI0Ii8+PC9zdmc+);
                padding: 2px;
            }
            QPlainTextEdit {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 5px;
                selection-background-color: #0078d4;
                selection-color: #ffffff;
                color: #000000;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 8px;
                margin: 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #c0c0c0;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                border: none;
                background: #f0f0f0;
                height: 8px;
                margin: 0px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #c0c0c0;
                min-width: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #a0a0a0;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
        """)
        
        # Set button names for specific styling
        self.start_button.setObjectName("start_button")
    
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
        """Open file dialog to select input directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Input Directory", 
            self.input_dir_edit.text() or str(Path.home())
        )
        if directory:
            self.input_dir_edit.setText(directory)
    
    def _browse_output_dir(self):
        """Open file dialog to select output directory."""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select Output Directory", 
            self.output_dir_edit.text() or str(Path.home())
        )
        if directory:
            self.output_dir_edit.setText(directory)
    
    def _start_transcription(self):
        """Start the transcription process."""
        if self.is_processing:
            logger.warning("Process already running")
            return
            
        # Validate inputs
        if not self._validate_inputs():
            return
            
        # Get settings from UI
        input_dir = self.input_dir_edit.text()
        output_dir = self.output_dir_edit.text()
        model_size = self.model_combo.currentText()
        
        # Get language code from selected language name
        language_name = self.language_combo.currentText()
        language = next((code for name, code in LANGUAGES if name == language_name), None)
        
        device = self.device_combo.currentText()
        
        # Override device if not using GPU
        if not self.use_gpu_check.isChecked() and device == "auto":
            device = "cpu"
            
        output_format = "txt" if self.format_txt_radio.isChecked() else "srt"
        skip_processed = self.skip_processed_check.isChecked()
        
        # Update UI state
        self.is_processing = True
        self.cancel_requested = False
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        
        # Reset progress bar
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting transcription...")
        
        # Start processing in a separate thread
        self.process_thread = threading.Thread(
            target=self._run_processing,
            args=(input_dir, output_dir, model_size, language, device, output_format, skip_processed)
        )
        self.process_thread.daemon = True
        self.process_thread.start()
    
    def _run_processing(self, input_dir, output_dir, model_size, language, device, output_format, skip_processed):
        """Run the transcription process in a background thread."""
        try:
            # Signal model loading started
            self.model_loading_progress.emit(10)
            
            # Define a progress callback to update the UI
            def progress_callback(current, total, filename=""):
                # Check if cancellation was requested
                if self.cancel_requested:
                    return False  # Signal to stop processing
                
                # Update progress
                self.progress_update.emit(current, total, filename)
                
                # Update current file progress (assume equal progress for each file)
                file_percentage = 0
                if "Transcribing" in filename:
                    # Update current file progress
                    file_percentage = 50  # Start at 50% when transcribing begins
                    self.current_file_progress.emit(file_percentage)
                elif "Processing" in filename:
                    # File is being processed
                    file_percentage = 25  # Start at 25% when processing begins
                    self.current_file_progress.emit(file_percentage)
                
                return True  # Continue processing
            
            # Signal file detection start
            self.file_detection_progress.emit(10)
            
            # Run the main processing function
            result = process_videos(
                input_path=input_dir,
                output_path=output_dir,
                model_size=model_size,
                language=language,
                device=device,
                output_format=output_format,
                max_workers=1,
                progress_callback=self._enhanced_progress_callback
            )
            
            if not self.cancel_requested:
                # Signal processing complete
                self.model_loading_progress.emit(100)
                self.file_detection_progress.emit(100)
                self.current_file_progress.emit(100)
                self.processing_complete.emit()
                logger.info(f"Processing complete. Processed {result.get('processed', 0)} files.")
            else:
                logger.info("Processing was cancelled by user.")
                
        except Exception as e:
            logger.error(f"Error in processing: {e}", exc_info=True)
            self.processing_error.emit("Processing Error", str(e))
            
    def _enhanced_progress_callback(self, current, total, filename=""):
        """Enhanced progress callback to update all progress indicators."""
        # Check if cancellation was requested
        if self.cancel_requested:
            return False  # Signal to stop processing
        
        # First, update the model loading progress at the beginning
        if current == 1 and total > 1:
            # Start of processing, model should be loaded
            self.model_loading_progress.emit(100)
            # File detection completed after the first file is found
            self.file_detection_progress.emit(100)
        
        # Update the main progress
        self.progress_update.emit(current, total, filename)
        
        # Extract the stage from the filename for detailed progress
        if isinstance(filename, str):
            # Check for specific keywords in the filename/status
            if "loading model" in filename.lower():
                self.model_loading_progress.emit(50)
            elif "detecting files" in filename.lower():
                self.file_detection_progress.emit(50)
            elif "transcribing" in filename.lower():
                # Increment file progress for transcription stage
                self.current_file_progress.emit(75)
            elif "processing" in filename.lower():
                # Initial file processing
                self.current_file_progress.emit(25)
            elif "complete" in filename.lower():
                # File completed
                self.current_file_progress.emit(100)
            
            # Reset current file progress when moving to a new file
            if current > 1 and "processing" in filename.lower():
                # Reset for new file
                self.current_file_progress.emit(0)
        
        return True  # Continue processing
    
    def _update_progress(self, current, total, filename):
        """Update progress bar and status label."""
        percentage = int((current / max(total, 1)) * 100)
        self.progress_bar.setValue(percentage)
        
        if filename:
            self.status_label.setText(f"Processing: {filename} ({current}/{total})")
        else:
            self.status_label.setText(f"Processing: {current}/{total}")
    
    def _on_processing_complete(self):
        """Handle processing complete event."""
        self.status_label.setText("Processing complete!")
        self.progress_bar.setValue(100)
        self.model_loading_progress.setValue(100)
        self.file_detection_progress.setValue(100)
        self.current_file_progress.setValue(100)
        self._reset_processing_state()
    
    def _show_error_dialog(self, title, message):
        """Show error dialog with details."""
        dialog = QtWidgets.QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        dialog.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
        dialog.exec()
    
    def _cancel_transcription(self):
        """Cancel the transcription process."""
        if not self.is_processing:
            return
            
        self.cancel_requested = True
        self.status_label.setText("Cancelling...")
        self.cancel_button.setEnabled(False)
    
    def _reset_processing_state(self):
        """Reset processing state after completion or cancellation."""
        self.is_processing = False
        self.cancel_requested = False
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        # Reset all progress indicators if cancellation occurred without completion
        if not self.progress_bar.value() == 100:
            self.progress_bar.setValue(0)
            self.model_loading_progress.setValue(0)
            self.file_detection_progress.setValue(0)
            self.current_file_progress.setValue(0)
            self.status_label.setText("Ready")
    
    def _validate_inputs(self):
        """Validate user inputs before starting processing."""
        input_dir = self.input_dir_edit.text()
        output_dir = self.output_dir_edit.text()
        
        if not input_dir:
            self._show_error_dialog("Error", "Please select an input directory.")
            return False
            
        if not output_dir:
            self._show_error_dialog("Error", "Please select an output directory.")
            return False
            
        if not os.path.isdir(input_dir):
            self._show_error_dialog("Error", f"Input directory does not exist: {input_dir}")
            return False
            
        if not os.path.isdir(output_dir):
            try:
                # Try to create the output directory if it doesn't exist
                os.makedirs(output_dir, exist_ok=True)
                logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                self._show_error_dialog("Error", f"Could not create output directory: {e}")
                return False
                
        return True
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.is_processing:
            # Ask for confirmation before closing
            confirm = QtWidgets.QMessageBox.question(
                self, "Confirm Exit",
                "A transcription is in progress. Are you sure you want to quit?",
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
                QtWidgets.QMessageBox.StandardButton.No
            )
            
            if confirm == QtWidgets.QMessageBox.StandardButton.Yes:
                # Cancel processing and close
                self.cancel_requested = True
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()

    def _create_app_icon(self):
        """Create a simple application icon programmatically."""
        # Create a pixmap for the icon
        icon_size = 64
        pixmap = QtGui.QPixmap(icon_size, icon_size)
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)
        
        # Create a painter to draw the icon
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        # Draw a blue rounded rectangle background
        painter.setBrush(QtGui.QColor("#0078d4"))
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.drawRoundedRect(2, 2, icon_size-4, icon_size-4, 12, 12)
        
        # Draw a white "W" in the center
        painter.setPen(QtGui.QPen(QtGui.QColor("#FFFFFF"), 3))
        painter.setFont(QtGui.QFont("Segoe UI", 36, QtGui.QFont.Weight.Bold))
        painter.drawText(QtCore.QRect(0, 0, icon_size, icon_size), QtCore.Qt.AlignmentFlag.AlignCenter, "W")
        
        # End painting
        painter.end()
        
        # Set the window icon
        self.setWindowIcon(QtGui.QIcon(pixmap))

    def _setup_overlap_detection(self):
        """Setup overlap detection for layout debugging."""
        # Create a timer to periodically check for overlaps
        self.overlap_timer = QtCore.QTimer(self)
        self.overlap_timer.setInterval(1000)  # Check every second
        self.overlap_timer.timeout.connect(self._check_widget_overlaps)
        self.overlap_timer.start()
    
    def _check_widget_overlaps(self):
        """Check if any widgets in the layout are overlapping."""
        if not hasattr(self, 'central_widget'):
            return
        
        # Get all visible widgets in the main layout
        widgets = []
        for i in range(self.main_layout.count()):
            widget = self.main_layout.itemAt(i).widget()
            if widget and widget.isVisible():
                widgets.append(widget)
        
        # Check for overlaps between widgets
        overlaps = []
        for i, widget1 in enumerate(widgets):
            for widget2 in widgets[i+1:]:
                if self._widgets_overlap(widget1, widget2):
                    overlaps.append((widget1.objectName() or type(widget1).__name__, 
                                     widget2.objectName() or type(widget2).__name__))
        
        # Log any overlaps found
        if overlaps:
            logger.warning(f"Widget overlaps detected: {overlaps}")
            
            # If scroll area is needed but scroll bars aren't visible, ensure they are
            if not self.scroll_area.horizontalScrollBar().isVisible() and not self.scroll_area.verticalScrollBar().isVisible():
                self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
                logger.info("Enabling scroll bars due to component overlap")
    
    def _widgets_overlap(self, widget1, widget2):
        """Check if two widgets overlap in the UI."""
        # Get global positions
        geo1 = widget1.geometry()
        geo2 = widget2.geometry()
        
        # Convert to global coordinates
        pos1 = widget1.mapToGlobal(QtCore.QPoint(0, 0))
        pos2 = widget2.mapToGlobal(QtCore.QPoint(0, 0))
        
        # Create QRects in global coordinates
        rect1 = QtCore.QRect(pos1.x(), pos1.y(), geo1.width(), geo1.height())
        rect2 = QtCore.QRect(pos2.x(), pos2.y(), geo2.width(), geo2.height())
        
        # Check for intersection
        return rect1.intersects(rect2)
    
    def resizeEvent(self, event):
        """Handle resize events to check if content fits in the window."""
        super().resizeEvent(event)
        
        # Get content size and available window size
        content_size = self.central_widget.sizeHint()
        window_size = self.size()
        
        # Check if content is larger than the window
        if content_size.height() > window_size.height() or content_size.width() > window_size.width():
            # Enable scroll bars
            self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            
            if content_size.height() > window_size.height():
                logger.debug(f"Content height ({content_size.height()}) exceeds window height ({window_size.height()}), enabling scroll bars")
        else:
            # Content fits, disable scroll bars
            self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    
    def showEvent(self, event):
        """Initialize UI state when the window is first shown."""
        super().showEvent(event)
        # Perform an initial check once all widgets are properly laid out
        QtCore.QTimer.singleShot(100, self._check_widget_overlaps)
        QtCore.QTimer.singleShot(100, lambda: self.resizeEvent(None))
        # Adjust window size based on content after UI is fully initialized
        QtCore.QTimer.singleShot(200, self._adjust_window_size)
    
    def _adjust_window_size(self):
        """Adjust the window size to fit all content optimally on startup."""
        # Get the size hint from central widget (ideal size to show everything)
        content_size = self.central_widget.sizeHint()
        
        # Add margins for window frame
        ideal_width = content_size.width() + 50
        ideal_height = content_size.height() + 50
        
        # Get current screen size
        screen = QtWidgets.QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        
        # Limit the window size to 90% of screen size
        max_width = int(screen_geometry.width() * 0.9)
        max_height = int(screen_geometry.height() * 0.9)
        
        # Calculate optimal size (not exceeding screen limits)
        optimal_width = min(ideal_width, max_width)
        optimal_height = min(ideal_height, max_height)
        
        # Only resize if content doesn't fit
        current_size = self.size()
        if optimal_width > current_size.width() or optimal_height > current_size.height():
            logger.debug(f"Adjusting window size to {optimal_width}x{optimal_height} to fit content")
            self.resize(optimal_width, optimal_height)
            
            # Center the window on screen
            frame_geometry = self.frameGeometry()
            center_point = screen_geometry.center()
            frame_geometry.moveCenter(center_point)
            self.move(frame_geometry.topLeft())

    def _update_model_loading_progress(self, percentage):
        """Update model loading progress wheel."""
        self.model_loading_progress.setValue(percentage)
        
    def _update_file_detection_progress(self, percentage):
        """Update file detection progress wheel."""
        self.file_detection_progress.setValue(percentage)
        
    def _update_current_file_progress(self, percentage):
        """Update current file progress wheel."""
        self.current_file_progress.setValue(percentage)

def main():
    """Main entry point for the Qt GUI application."""
    app = QtWidgets.QApplication(sys.argv)
    
    # Set up application-wide font
    app.setFont(QtGui.QFont("Segoe UI", 9))
    
    # Apply Windows 11 style if on Windows
    if sys.platform == "win32":
        try:
            from qtpy.QtWidgets import QStyleFactory
            app.setStyle(QStyleFactory.create("Fusion"))
        except Exception as e:
            logger.warning(f"Could not set Windows 11 style: {e}")
    
    window = WhisperBatchQt()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 