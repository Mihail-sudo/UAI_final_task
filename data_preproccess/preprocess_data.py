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

FRAME_LENGTH = 2048
FRAME_STEP = 512

SQRT_HANN_WINDOW = tf.sqrt(
    tf.signal.hann_window(
        FRAME_LENGTH,
        periodic=True
    )
)

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


def pad_spectrogram(mag):
    """
        Расширить размер аудио до необходимого для корректной работы модели
    """
    h, w = mag.shape[:2]

    pad_h = (CONV_SIZE - h % CONV_SIZE) % CONV_SIZE
    pad_w = (CONV_SIZE - w % CONV_SIZE) % CONV_SIZE

    mag = np.pad(mag, 
        [
            (0, pad_h),
            (0, pad_w),
            (0, 0)
        ], 
        mode="constant"
    )
    return mag


def sqrt_hann_window(frame_length, dtype):
    return tf.cast(SQRT_HANN_WINDOW, dtype)


def to_stft(audio):
    """
        Waveform -> Complex STFT
    """

    return tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FRAME_LENGTH,
        window_fn=sqrt_hann_window
    )


def inverse_stft(stft):
    """
    Complex STFT -> Waveform
    """

    return tf.signal.inverse_stft(
        stft,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FRAME_LENGTH,
        window_fn=sqrt_hann_window
    )


def stft_to_mag(stft, log_scale=True):
    mag = tf.abs(stft)

    if log_scale:
        mag = tf.math.abs(mag)

    return mag[..., tf.newaxis].numpy().astype(np.float32)


def stft_to_phase(stft):
    return tf.math.angle(stft).numpy().astype(np.float16)


def random_chunk(*tracks):
    """
        обрезает треки до необходимого кол-ва секунд
    """
    length = len(tracks[0])

    if length <= CHUNK_SAMPLES:
        result = []
        for track in tracks:
            pad = CHUNK_SAMPLES - len(track)
            result.append(np.pad(track, (0, pad), mode="constant"))
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
        Обработка песни: из одного стема получается NUM_CHUNKS CHUNK_SEC секундных трека
    """

    mix, drums, bass, other, vocals = load_song(input_path)

    name = os.path.basename(input_path)
    name = name.split('.')[0]
    for i in range(NUM_CHUNKS):
        mix_chunk, drums_chunk, bass_chunk, other_chunk, vocals_chunk = random_chunk(mix, drums, bass, other, vocals)
        
        mix_stft = to_stft(mix_chunk)
        mix_mag = pad_spectrogram(stft_to_mag(mix_stft))
        # mix_phase = pad_spectrogram(stft_to_phase(mix_stft)[..., np.newaxis])

        drums_stft = to_stft(drums_chunk)
        drums_mag = pad_spectrogram(stft_to_mag(drums_stft))

        bass_stft = to_stft(bass_chunk)
        bass_mag = pad_spectrogram(stft_to_mag(bass_stft))

        other_stft = to_stft(other_chunk)
        other_mag = pad_spectrogram(stft_to_mag(other_stft))

        vocals_stft = to_stft(vocals_chunk)
        vocals_mag = pad_spectrogram(stft_to_mag(vocals_stft))


        eps = 1e-8

        drums_mask = drums_mag / (mix_mag + eps)
        bass_mask = bass_mag / (mix_mag + eps)
        other_mask = other_mag / (mix_mag + eps)
        vocals_mask = vocals_mag / (mix_mag + eps)

        np.savez_compressed(
            os.path.join(output_dir, f"{name}_{i:02d}.npz"),
            # mix_phase=mix_phase,
            mix=mix_mag,
            drums=drums_mask,
            bass=bass_mask,
            other=other_mask,
            vocals=vocals_mask,
        )