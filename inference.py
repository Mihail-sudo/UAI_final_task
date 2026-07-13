import os
import numpy as np
import soundfile as sf
import tensorflow as tf

# Must match dataset.py MR_STFT_CONFIGS
MR_STFT_CONFIGS = [
    (2048, 512),   # reference
    (1024, 256),
]
CONV_SIZE = 16

CHUNK_FRAMES = 864  # ~10s at 44100 Hz / 512 stride
CHUNK_HOP = CHUNK_FRAMES // 2


def _make_window_fn(frame_length):
    window = tf.sqrt(tf.signal.hann_window(frame_length, periodic=True))
    def _fn(fl, dtype):
        return tf.cast(window, dtype)
    return _fn


def load_audio(path, sr=44100):
    audio, orig_sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if orig_sr != sr:
        audio = tf.signal.resample(audio, orig_sr, sr).numpy()
    return audio


def pad_spectrogram(spec):
    h, w = spec.shape[:2]
    pad_h = (CONV_SIZE - h % CONV_SIZE) % CONV_SIZE
    pad_w = (CONV_SIZE - w % CONV_SIZE) % CONV_SIZE
    if pad_h == 0 and pad_w == 0:
        return spec
    pad_width = [(0, pad_h), (0, pad_w)]
    for _ in range(spec.ndim - 2):
        pad_width.append((0, 0))
    return np.pad(spec, pad_width, mode="constant")


def separate(audio, model, configs=None):
    if configs is None:
        configs = MR_STFT_CONFIGS

    window_fns = [_make_window_fn(fl) for fl, _ in configs]

    # Reference STFT config (first) — used for phase and ISTFT
    ref_fl, ref_fs = configs[0]
    ref_wfn = window_fns[0]

    ref_stft = tf.signal.stft(audio, frame_length=ref_fl, frame_step=ref_fs, fft_length=ref_fl, window_fn=ref_wfn)
    ref_mag = tf.abs(ref_stft).numpy()
    ref_phase = tf.math.angle(ref_stft).numpy()
    T, F = ref_mag.shape

    # Build multi-resolution input stack
    channels = []
    for i, ((fl, fs), wfn) in enumerate(zip(configs, window_fns)):
        s = tf.signal.stft(audio, frame_length=fl, frame_step=fs, fft_length=fl, window_fn=wfn)
        mag = tf.abs(s).numpy()
        log_mag = np.log1p(mag)

        if i == 0:
            channels.append(log_mag)
        else:
            log_mag = tf.image.resize(log_mag[np.newaxis, :, :, np.newaxis], [T, F]).numpy()
            channels.append(log_mag[0, :, :, 0])

    spec = np.stack(channels, axis=-1).astype(np.float32)  # (T, F, N)

    padded_spec = pad_spectrogram(spec)
    T_pad, F_pad = padded_spec.shape[:2]

    mask_accum = np.zeros((T_pad, F_pad, 4), dtype=np.float64)
    weight_accum = np.zeros(T_pad, dtype=np.float64)
    ola_window = np.hanning(CHUNK_FRAMES)

    for start in range(0, T_pad, CHUNK_HOP):
        end = start + CHUNK_FRAMES
        chunk = padded_spec[start:end, :, :]

        pad = 0
        if chunk.shape[0] < CHUNK_FRAMES:
            pad = CHUNK_FRAMES - chunk.shape[0]
            chunk = np.pad(chunk, [(0, pad), (0, 0), (0, 0)])

        chunk_input = chunk[np.newaxis, :, :, :]  # (1, t, f, N)
        masks = model.predict(chunk_input, verbose=0)[0]

        if pad:
            masks = masks[:-pad]

        n = masks.shape[0]
        mask_accum[start : start + n] += masks * ola_window[:n, np.newaxis, np.newaxis]
        weight_accum[start : start + n] += ola_window[:n]

    mask_accum /= weight_accum[:, np.newaxis, np.newaxis] + 1e-8

    masks = mask_accum[:T, :F, :].astype(np.float32)

    # ISTFT reconstruction with reference phase
    sources = []
    for i in range(4):
        source_stft = masks[:, :, i] * ref_mag * np.exp(1j * ref_phase)
        source_wav = tf.signal.inverse_stft(
            source_stft,
            frame_length=ref_fl,
            frame_step=ref_fs,
            fft_length=ref_fl,
            window_fn=ref_wfn,
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
