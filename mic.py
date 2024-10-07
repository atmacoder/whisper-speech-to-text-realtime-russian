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
    # Удаление всего текста в квадратных скобках (например, [музыка], [шум])
    text = re.sub(r'\[.*?\]', '', text)
    # Дополнительная очистка от лишних пробелов
    return text.strip()


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

    phrase_time = None
    data_queue = Queue()
    recorder = sr.Recognizer()
    recorder.energy_threshold = args.energy_threshold
    recorder.dynamic_energy_threshold = False

    # Выбор микрофона
    if 'linux' in platform:
        mic_name = args.default_microphone
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    # Загрузка модели Whisper
    model = args.model
    audio_model = whisper.load_model(model)

    record_timeout = args.record_timeout
    transcription = ['']

    # Настройка микрофона
    with source:
        recorder.adjust_for_ambient_noise(source)

    # Создание папки output и генерация случайного файла для записи
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    random_filename = generate_random_filename()
    filepath = os.path.join(output_dir, random_filename)

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to receive audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        data = audio.get_raw_data()
        data_queue.put(data)

    # Запуск записи в фоновом режиме
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print(f"Model loaded. Transcription will be saved in {filepath}\n")

    while True:
        try:
            now = datetime.utcnow()
            if not data_queue.empty():
                # Получение данных из очереди
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()

                # Преобразование в формат, который понимает модель
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Транскрибирование аудио с указанием русского языка
                result = audio_model.transcribe(audio_np, language="ru", fp16=torch.cuda.is_available())
                text = result['text'].strip()

                # Очистка текста от меток в квадратных скобках и лишних пробелов
                cleaned_text = clean_transcription(text)

                # Если есть текст после очистки, то записываем его в файл
                if cleaned_text:
                    append_transcription_to_file(cleaned_text, filepath)
                    transcription.append(cleaned_text)

                # Очистка консоли и вывод текущей транскрипции
                os.system('cls' if os.name == 'nt' else 'clear')
                for line in transcription:
                    print(line)
                print('', end='', flush=True)
            else:
                sleep(0.25)  # Задержка для предотвращения перегрузки процессора
        except KeyboardInterrupt:
            break

    print(f"\nTranscription saved to {filepath}")


if __name__ == "__main__":
    main()
