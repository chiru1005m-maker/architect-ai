@echo off
title ARCHITECT-AI LAUNCHER
echo [1/3] Checking Ollama Status...
start /b ollama serve
timeout /t 3 >nul

echo [2/3] Verifying Model Residency...
ollama pull deepseek-r1:7b
ollama pull qwen2.5-coder:1.5b

echo [3/3] Launching Titanium Interface...
streamlit run main.py
pause
