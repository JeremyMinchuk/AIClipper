"""
AI Video Editor - Quick Start Script
Run this file to launch the application
"""
import subprocess
import sys
import os


def check_dependencies():
    """Check if required packages are installed"""
    required = ['gradio', 'moviepy', 'whisper', 'ollama', 'PIL', 'cv2', 'torch']
    missing = []

    for package in required:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'whisper':
                import whisper
            else:
                __import__(package)
        except ImportError:
            missing.append(package)

    return missing


def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
    ])


def main():
    print("=" * 60)
    print("🎬 AI Video Editor - Startup")
    print("=" * 60)

    # Check dependencies
    print("\n📦 Checking dependencies...")
    missing = check_dependencies()

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        response = input("Install dependencies now? (y/n): ")
        if response.lower() == 'y':
            install_dependencies()
        else:
            print("Please install dependencies manually:")
            print("  pip install -r requirements.txt")
            return

    # Import and run the app
    print("\n🚀 Launching AI Video Editor...")
    print("   The browser will open automatically.")
    print("   Press Ctrl+C to stop the server.\n")

    from app import main as run_app
    run_app()


if __name__ == "__main__":
    main()
