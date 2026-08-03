import os
import numpy as np
import soundfile as sf
import tensorflow as tf
from refine import create_refine_model

# Must match dataset.py MR_STFT_CONFIGS
MR_STFT_CONFIGS = [
    (4096, 2048),  # reference — used for mask computation & ISTFT
    (8192, 4096),  # high frequency resolution
    (2048, 1024),  # mid frequency resolution
    (1024, 512),   # high time resolution
    (256, 128),    # very high time resolution
]
CONV_SIZE = 32

CHUNK_FRAMES = 224  # ~10.4s at 44100 Hz / 2048 stride, divisible by CONV_SIZE=32
CHUNK_HOP = CHUNK_FRAMES // 2


EPS = 1e-8


def si_sdr(estimate, reference):
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + EPS)
    noise = estimate - alpha * reference
    sdr = 10 * np.log10(
        np.dot(alpha * reference, alpha * reference) / (np.dot(noise, noise) + EPS) + EPS
    )
    return float(sdr)


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


def median_filter_3x3(x):
    """3x3 median filter applied per channel. x: (T, F, C) -> (T, F, C)."""
    T = tf.shape(x)[0]
    F = tf.shape(x)[1]
    padded = tf.pad(x, [[1, 1], [1, 1], [0, 0]], mode="REFLECT")
    neighbors = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbors.append(padded[1 + dy:1 + dy + T, 1 + dx:1 + dx + F])
    stacked = tf.stack(neighbors, axis=-1)  # (T, F, C, 9)
    return tf.sort(stacked, axis=-1)[..., 4]


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


def separate(audio, model, configs=None, stats_file=None, refine_model=None):
    if configs is None:
        configs = MR_STFT_CONFIGS

    if stats_file is not None:
        stats = np.load(stats_file)
        norm_mean = stats["mean"].astype(np.float32)
        norm_std = stats["std"].astype(np.float32)
    else:
        norm_mean = norm_std = None

    window_fns = [_make_window_fn(fl) for fl, _ in configs]

    # Reference STFT config (first) — used for ISTFT
    ref_fl, ref_fs = configs[0]
    ref_wfn = window_fns[0]

    ref_stft = tf.signal.stft(audio, frame_length=ref_fl, frame_step=ref_fs, fft_length=ref_fl, window_fn=ref_wfn).numpy()
    T, F = ref_stft.shape

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

    if norm_mean is not None:
        spec = (spec - norm_mean) / (norm_std + 1e-8)

    padded_spec = pad_spectrogram(spec)
    T_pad, F_pad = padded_spec.shape[:2]

    mask_accum = np.zeros((T_pad, F_pad, 8), dtype=np.float64)
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
        # direct call instead of model.predict: predict() builds its own tf.data
        # pipeline, which deadlocks when called from inside a dataset generator
        pred = model(chunk_input, training=False).numpy()[0]

        if pad:
            pred = pred[:-pad]

        n = pred.shape[0]
        mask_accum[start : start + n] += pred * ola_window[:n, np.newaxis, np.newaxis]
        weight_accum[start : start + n] += ola_window[:n]

    mask_accum /= weight_accum[:, np.newaxis, np.newaxis] + 1e-8

    pred = mask_accum[:T, :F, :].astype(np.float32)  # (T, F, 8)

    # 3x3 median smoothing: removes speckle in noise-floor bins without
    # cutting any frequencies
    pred = median_filter_3x3(pred).numpy()

    mask_r = pred[:, :, :4]  # (T, F, 4)
    mask_i = pred[:, :, 4:]  # (T, F, 4)

    # ISTFT reconstruction with complex masking
    sources = []
    for i in range(4):
        source_stft = (mask_r[:, :, i] + 1j * mask_i[:, :, i]) * ref_stft
        source_wav = tf.signal.inverse_stft(
            source_stft,
            frame_length=ref_fl,
            frame_step=ref_fs,
            fft_length=ref_fl,
            window_fn=ref_wfn,
        ).numpy()
        source_wav = source_wav[: len(audio)]
        if refine_model is not None:
            inp = source_wav.reshape(1, -1, 1).astype(np.float32)
            source_wav = refine_model.predict(inp, verbose=0)[0, :, 0]
            source_wav = source_wav[: len(audio)]
        sources.append(source_wav)

    return np.stack(sources, axis=-1)


def separate_file(input_path, output_dir, model, stats_file=None, refine_model=None, ref_dir=None):
    audio = load_audio(input_path)
    stems = separate(audio, model, stats_file=stats_file, refine_model=refine_model)

    names = ["drums", "bass", "other", "vocals"]
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]

    for i, name in enumerate(names):
        path = os.path.join(output_dir, f"{base}_{name}.wav")
        sf.write(path, stems[:, i], 44100)
        print(f"Saved: {path}")

    if ref_dir is not None:
        sdrs = []
        for i, name in enumerate(names):
            ref_path = os.path.join(ref_dir, f"{name}.wav")
            if not os.path.exists(ref_path):
                print(f"Reference not found: {ref_path}")
                continue
            ref = load_audio(ref_path)
            T = min(len(stems[:, i]), len(ref))
            sdr = si_sdr(stems[:T, i], ref[:T])
            sdrs.append(sdr)
            print(f"  SI-SDR ({name:>6s}): {sdr:.2f} dB")
        if sdrs:
            print(f"  SI-SDR (avg): {np.mean(sdrs):.2f} dB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to input audio file")
    parser.add_argument("--output-dir", "-o", default="separated")
    parser.add_argument("--model", "-m", default="separator.keras")
    parser.add_argument("--refine", "-r", default="refine.keras",
                        help="Path to refine model (skip if not found)")
    parser.add_argument("--stats", "-s", default="musdb18/stats.npz",
                        help="Path to stats.npz for input normalization (omit to skip normalization)")
    parser.add_argument("--ref-dir", default=None,
                        help="Directory with reference drums/bass/other/vocals.wav to compute SI-SDR")
    args = parser.parse_args()

    if not os.path.exists(args.stats):
        print(f"Stats file not found at {args.stats}, skipping normalization")
        args.stats = None

    refine_model = None
    if args.refine and os.path.exists(args.refine):
        refine_model = tf.keras.models.load_model(args.refine)
        print(f"Loaded refine model: {args.refine}")

    model = tf.keras.models.load_model(args.model)
    separate_file(args.input, args.output_dir, model, stats_file=args.stats,
                  refine_model=refine_model, ref_dir=args.ref_dir)
