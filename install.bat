@echo off
echo Установка зависимостей...

pip install numpy
pip install SpeechRecognition
pip install openai-whisper
pip install torch
pip install pyaudio

echo Зависимости установлены.
pause
