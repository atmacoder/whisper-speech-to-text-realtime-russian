import pyaudio
import numpy as np
import whisper
import torch
import os
import random
import string
import re

def generate_random_filename(extension=".txt"):
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return random_string + extension

def append_transcription_to_file(text, filepath):
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(text + '\n')

def clean_transcription(text):
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()

samplerate = 44100  # Частота дискретизации
blocksize = 1024  # Размер блока данных
channels = 2  # Количество каналов (стерео)

# Инициализация PyAudio
p = pyaudio.PyAudio()

# Поиск устройства "Stereo Mix" или "What-U-Hear"
device_index = None
for i in range(p.get_device_count()):
    dev_info = p.get_device_info_by_index(i)
    if "Stereo Mix" in dev_info['name'] or "What-U-Hear" in dev_info['name']:
        device_index = i
        print(f"Using device: {dev_info['name']} (index {i}) for capturing playback")
        break

if device_index is None:
    print("No suitable Stereo Mix or What-U-Hear device found.")
    exit()

# Загрузка модели Whisper
model = "medium"
audio_model = whisper.load_model(model)

# Создание папки output и генерация случайного файла для записи
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

random_filename = generate_random_filename()
filepath = os.path.join(output_dir, random_filename)

# Функция для обработки аудиопотока
def process_audio(data, frames):
    audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    result = audio_model.transcribe(audio_np, language="ru", fp16=torch.cuda.is_available())
    text = result['text'].strip()
    cleaned_text = clean_transcription(text)
    if cleaned_text:
        append_transcription_to_file(cleaned_text, filepath)
        print(cleaned_text)

# Запуск потока записи
try:
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=samplerate,
                    input=True,
                    frames_per_buffer=blocksize,
                    input_device_index=device_index)
    print(f"Recording... Transcription will be saved in {filepath}")
    while True:
        data = stream.read(blocksize, exception_on_overflow=False)
        process_audio(data, blocksize)
except KeyboardInterrupt:
    print("Recording stopped.")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()

print(f"\nTranscription saved to {filepath}")
