import os
import numpy as np
import soundfile as sf
import tensorflow as tf

from model import create_unet

FRAME_LENGTH = 2048
FRAME_STEP = 512
CONV_SIZE = 16

SQRT_HANN_WINDOW = tf.sqrt(
    tf.signal.hann_window(FRAME_LENGTH, periodic=True)
)


def sqrt_hann_window(frame_length, dtype):
    return tf.cast(SQRT_HANN_WINDOW, dtype)


def load_audio(path, sr=44100):
    audio, orig_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if orig_sr != sr:
        audio = tf.signal.resample(audio, orig_sr, sr).numpy()
    return audio


def pad_spectrogram(mag):
    h, w = mag.shape[:2]
    pad_h = (CONV_SIZE - h % CONV_SIZE) % CONV_SIZE
    pad_w = (CONV_SIZE - w % CONV_SIZE) % CONV_SIZE
    if pad_h == 0 and pad_w == 0:
        return mag
    return np.pad(mag, [(0, pad_h), (0, pad_w)], mode="constant")


def separate(audio, model):
    stft = tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FRAME_LENGTH,
        window_fn=sqrt_hann_window,
    )

    mag = tf.abs(stft).numpy()
    phase = tf.math.angle(stft).numpy()

    padded_mag = pad_spectrogram(mag)
    log_mag = np.log1p(padded_mag)
    log_mag = log_mag[np.newaxis, :, :, np.newaxis]

    masks = model.predict(log_mag, verbose=0)[0]

    h, w = mag.shape
    masks = masks[:h, :w]

    sources = []
    for i in range(4):
        source_stft = masks[:, :, i] * mag * np.exp(1j * phase)
        source_wav = tf.signal.inverse_stft(
            source_stft,
            frame_length=FRAME_LENGTH,
            frame_step=FRAME_STEP,
            fft_length=FRAME_LENGTH,
            window_fn=sqrt_hann_window,
        ).numpy()
        source_wav = source_wav[: len(audio)]
        sources.append(source_wav)

    return np.stack(sources, axis=-1)


def separate_file(input_path, output_dir, model):
    audio = load_audio(input_path)
    stems = separate(audio, model)

    names = ["drums", "bass", "other", "vocals"]
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]

    for i, name in enumerate(names):
        path = os.path.join(output_dir, f"{base}_{name}.wav")
        sf.write(path, stems[:, i], 44100)
        print(f"Saved: {path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("--output-dir", "-o", default="separated")
    parser.add_argument("--model", "-m", default="separator.keras")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    separate_file(args.input, args.output_dir, model)
