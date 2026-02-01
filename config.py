"""
Configuration settings for AI Video Editor
"""
import os

# Ollama Settings
OLLAMA_MODEL = "artifish/llama3.2-uncensored:latest"
OLLAMA_HOST = "http://localhost:11434"

# Video Processing Settings
OUTPUT_DIR = "output"
TEMP_DIR = "temp"
SUPPORTED_FORMATS = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

# Short-form video settings
SHORT_MIN_DURATION = 15  # seconds
SHORT_MAX_DURATION = 60  # seconds
SHORT_ASPECT_RATIO = (9, 16)  # vertical video
SHORT_RESOLUTION = (1080, 1920)  # width x height

# Caption Settings
CAPTION_FONT_SIZE = 48
CAPTION_FONT_COLOR = "white"
CAPTION_HIGHLIGHT_COLOR = "yellow"
CAPTION_BG_COLOR = (0, 0, 0, 180)  # RGBA - semi-transparent black
CAPTION_POSITION = "bottom"  # bottom, center, top
CAPTION_MARGIN = 100  # pixels from edge

# Whisper Settings
WHISPER_MODEL = "base"  # tiny, base, small, medium, large

# Processing Settings
MAX_CLIPS_TO_GENERATE = 10
OVERLAP_THRESHOLD = 0.5  # seconds - minimum gap between clips

# Create directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
