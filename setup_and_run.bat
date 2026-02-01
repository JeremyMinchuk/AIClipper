@echo off
echo ================================================
echo AI Video Editor - Setup and Run
echo ================================================
echo.

cd /d "%~dp0"

echo Step 1: Fixing package conflicts...
pip uninstall numpy opencv-python opencv-python-headless ollama -y 2>nul

echo.
echo Step 2: Installing dependencies...
pip install numpy==1.24.3
pip install opencv-python==4.8.0.74
pip install moviepy==1.0.3
pip install gradio==4.19.2
pip install openai-whisper
pip install Pillow requests tqdm ffmpeg-python

echo.
echo Step 3: Checking Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Ollama is not running!
    echo Please start Ollama in another terminal with: ollama serve
    echo Then pull the model with: ollama pull artifish/llama3.2-uncensored
    echo.
    pause
)

echo.
echo Step 4: Starting AI Video Editor...
python app.py

pause
