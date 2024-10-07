import argparse
import os
import re  # Для удаления текста в квадратных скобках
import numpy as np
import speech_recognition as sr
import whisper
import torch
import random
import string

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform


def generate_random_filename(extension=".txt"):
    """ Генерация случайного имени файла """
    random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
    return random_string + extension


def append_transcription_to_file(text, filepath):
    """ Добавление новой строки транскрипции в файл """
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(text + '\n')


def clean_transcription(text):
    """ Очистка текста от меток и нежелательных фрагментов """
    text = re.sub(r'\[.*?\]', '', text)
    return text.strip()


def list_supported_audio_devices():
    """ Возвращает список поддерживаемых аудиоустройств (тех, которые можно открыть для записи) """
    supported_devices = []
    all_devices = sr.Microphone.list_microphone_names()

    print("Checking available audio devices...")

    for index, name in enumerate(all_devices):
        try:
            mic = sr.Microphone(device_index=index)
            with mic as source:
                if hasattr(source, 'stream') and source.stream is not None:
                    supported_devices.append((index, name))
        except OSError as e:
            print(f"Skipping device {name} (index {index}) due to error: {str(e)}")
        except AttributeError:
            print(f"Skipping device {name} (index {index}) due to unsupported operation.")

    return supported_devices


def select_audio_device():
    """ Позволяет пользователю выбрать поддерживаемое устройство для записи """
    supported_devices = list_supported_audio_devices()

    if not supported_devices:
        print("No supported audio devices found.")
        return None

    print("Available supported audio devices:")
    for index, (device_index, device_name) in enumerate(supported_devices):
        print(f"{index}: {device_name}")

    user_choice = int(input("Select the device index you want to use: "))
    
    if user_choice >= 0 and user_choice < len(supported_devices):
        return supported_devices[user_choice][0]
    else:
        print("Invalid selection.")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large"])
    parser.add_argument("--non_english", action='store_true',
                        help="Don't use the English model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)
    
    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    device_index = select_audio_device()

    if device_index is None:
        print("No valid device selected. Exiting.")
        return

    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    try:
        source = sr.Microphone(device_index=device_index, sample_rate=16000)

        # Проверяем, поддерживает ли устройство запись
        with source as test_source:
            if test_source.stream is None:
                print("Selected device does not support recording. Please select another device.")
                return
        
        # Настраиваем шумоподавление
        with source:
            recorder.adjust_for_ambient_noise(source)
        
        # Загрузка модели Whisper
        model = args.model
        audio_model = whisper.load_model(model)

        record_timeout = args.record_timeout
        transcription = ['']

        # Создание папки output и генерация случайного файла для записи
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        random_filename = generate_random_filename()
        filepath = os.path.join(output_dir, random_filename)

        def record_callback(_, audio: sr.AudioData) -> None:
            data = audio.get_raw_data()
            data_queue.put(data)

        # Запуск записи в фоновом режиме
        recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

        print(f"Model loaded. Transcription will be saved in {filepath}\n")

        while True:
            try:
                now = datetime.utcnow()
                if not data_queue.empty():
                    audio_data = b''.join(data_queue.queue)
                    data_queue.queue.clear()

                    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                    result = audio_model.transcribe(audio_np, language="ru", fp16=torch.cuda.is_available())
                    text = result['text'].strip()

                    cleaned_text = clean_transcription(text)

                    if cleaned_text:
                        append_transcription_to_file(cleaned_text, filepath)
                        transcription.append(cleaned_text)

                    os.system('cls' if os.name == 'nt' else 'clear')
                    for line in transcription:
                        print(line)
                    print('', end='', flush=True)
                else:
                    sleep(0.25)
            except KeyboardInterrupt:
                break
    except OSError as e:
        print(f"Error: {e}")
        print("Please select another device.")
    finally:
        if source.stream:
            source.stream.close()

    print(f"\nTranscription saved to {filepath}")


if __name__ == "__main__":
    main()
