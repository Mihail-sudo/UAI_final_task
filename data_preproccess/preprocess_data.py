import soundfile as sf
import ffmpeg
import numpy as np
import tensorflow as tf
import random

import io
import os

# Частота дискретизации
GLOBAL_RATE = 44100

# длинна аудио
CHUNK_SEC = 10
NUM_CHUNKS = 10
# Кол-во сэмплов для обучния
CHUNK_SAMPLES = GLOBAL_RATE * CHUNK_SEC

FRAME_LENGTH = 1024
FRAME_STEP = 256

# параметр отвечающий за размер свертки / развертки модели (нужен для корректной работы encoder / decoder)
CONV_SIZE = 16


def extract_audio_to_memory(input_file, track_index='0:a:0'):
    """
    Вытащить нужный stem из .wav/.stem.m4a в память (bytes)
    """
    process = (
        ffmpeg
        .input(input_file)
        .output('pipe:', format='wav', acodec='pcm_s16le', ar=44100, ac=1, map=track_index)
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    out, err = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(err.decode())
    return out


def load_audio_from_memory(audio_bytes, mono=True):   
    """
        Преобразовать Аудио в waveform / sample rate
    """ 
    audio_buffer = io.BytesIO(audio_bytes)
    waveform, sample_rate = sf.read(audio_buffer, dtype="float32")
    
    if waveform.ndim > 1 and mono:
        waveform = np.mean(waveform, axis=1)

    if len(waveform.shape) == 2:
        waveform = waveform.T
    return waveform, sample_rate


def load_audio(input_file, track_index='0:a:0', mono=True):
    """
        Загрузить нужный стем канал в формате waveform / sample rate
    """
    out = extract_audio_to_memory(input_file, track_index)
    mix_waveform, sr = load_audio_from_memory(out, mono)
    return mix_waveform, sr


def pad_mag(mag):
    """
        Расширить размер аудио до необходимого для корректной работы модели
    """
    h, w = mag.shape[:2]

    pad_h = (CONV_SIZE - h % CONV_SIZE) % CONV_SIZE
    pad_w = (CONV_SIZE - w % CONV_SIZE) % CONV_SIZE

    mag = tf.pad(mag, 
        [
            [0, pad_h],
            [0, pad_w],
            [0, 0]
        ]
    )
    return mag.numpy()


def to_mag(audio):
    """
        Преобразует waveform в magnitude
    """
    stft = tf.signal.stft(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP)

    mag = tf.abs(stft)
    mag = mag[..., tf.newaxis]

    return mag.numpy()


def random_chunk(*tracks):
    """
        обрезает треки до необходимого кол-ва секунд
    """
    length = len(tracks[0])

    if length <= CHUNK_SAMPLES:
        result = []
        for track in tracks:
            pad = CHUNK_SAMPLES - len(track)
            result.append(np.pad(track, (0, pad)))
        return result

    start = random.randint(0, length - CHUNK_SAMPLES)
    end = start + CHUNK_SAMPLES

    return [track[start:end] for track in tracks]


def load_song(input_path):
    """
        Вспомогательная функция: загрузка всех каналов
        0:a:0 mix
        0:a:1 drums
        0:a:2 bass
        0:a:3 other
        0:a:4 vocals
    """
    mix, _ = load_audio(input_path, "0:a:0")
    drums, _ = load_audio(input_path, "0:a:1")
    bass, _ = load_audio(input_path, "0:a:2")
    other, _ = load_audio(input_path, "0:a:3")
    vocals, _ = load_audio(input_path, "0:a:4")

    return mix, drums, bass, other, vocals


def preprocess_song(input_path, output_dir):
    """
        Обработка песни: из одного стема получается 20 20-ти секундных трека
    """

    mix, drums, bass, other, vocals = load_song(input_path)

    name = os.path.basename(input_path)
    name = name.split('.')[0]
    for i in range(NUM_CHUNKS):
        mix_chunk, drums_chunk, bass_chunk, other_chunk, vocals_chunk = random_chunk(mix, drums, bass, other, vocals)
        
        mix_mag = pad_mag(to_mag(mix_chunk)).astype(np.float32)
        drums_mag = pad_mag(to_mag(drums_chunk)).astype(np.float32)
        bass_mag = pad_mag(to_mag(bass_chunk)).astype(np.float32)
        other_mag = pad_mag(to_mag(other_chunk)).astype(np.float32)
        vocals_mag = pad_mag(to_mag(vocals_chunk)).astype(np.float32)

        np.savez_compressed(
            os.path.join(output_dir, f"{name}_{i:02d}.npz"),
            mix=mix_mag,
            drums=drums_mag,
            bass=bass_mag,
            other=other_mag,
            vocals=vocals_mag,
        )