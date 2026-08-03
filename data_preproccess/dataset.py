import os
from functools import partial

import numpy as np
import tensorflow as tf

# ==========================================================
# CONFIG
# ==========================================================

# Multi-resolution STFT configs: list of (frame_length, frame_step)
# First entry is the reference resolution used for mask computation
MR_STFT_CONFIGS = [
    (4096, 2048),  # reference — used for mask computation & ISTFT
    (8192, 4096),  # high frequency resolution
    (2048, 1024),  # mid frequency resolution
    (1024, 512),   # high time resolution
    (256, 128),    # very high time resolution
]
CONV_SIZE = 32
EPS = 1e-4


def _make_window_fn(frame_length):
    window = tf.sqrt(tf.signal.hann_window(frame_length, periodic=True))
    def _fn(fl, dtype):
        return tf.cast(window, dtype)
    return _fn


# ==========================================================
# TFRecord schema
# ==========================================================

FEATURE_DESCRIPTION = {
    "length": tf.io.FixedLenFeature([], tf.int64),

    "mix": tf.io.FixedLenFeature([], tf.string),
    "drums": tf.io.FixedLenFeature([], tf.string),
    "bass": tf.io.FixedLenFeature([], tf.string),
    "other": tf.io.FixedLenFeature([], tf.string),
    "vocals": tf.io.FixedLenFeature([], tf.string),
}


# ==========================================================
# Parse TFRecord
# ==========================================================

def parse_tfrecord(example_proto):
    example = tf.io.parse_single_example(
        example_proto,
        FEATURE_DESCRIPTION
    )

    length = tf.cast(example["length"], tf.int32)

    def decode(key):
        x = tf.io.decode_raw(example[key], tf.float32)
        x = tf.reshape(x, [length])
        return x

    return {
        "mix": decode("mix"),
        "drums": decode("drums"),
        "bass": decode("bass"),
        "other": decode("other"),
        "vocals": decode("vocals"),
    }


# ==========================================================
# Padding
# ==========================================================

def pad_feature(x):
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]

    pad_h = (CONV_SIZE - h % CONV_SIZE) % CONV_SIZE
    pad_w = (CONV_SIZE - w % CONV_SIZE) % CONV_SIZE

    paddings = [[0, pad_h], [0, pad_w]]

    ndims = x.shape.rank
    if ndims is not None and ndims > 2:
        paddings.extend([[0, 0]] * (ndims - 2))

    return tf.pad(x, paddings)


# ==========================================================
# Magnitude
# ==========================================================

def magnitude(stft):
    return tf.abs(stft)


def log_magnitude(mag):
    return tf.math.log1p(mag)


# ==========================================================
# Spectrogram
# ==========================================================

def waveform_to_features(audio, configs=None):
    if configs is None:
        configs = MR_STFT_CONFIGS

    ref_fl, ref_fs = configs[0]
    ref_wfn = _make_window_fn(ref_fl)
    ref_stft = tf.signal.stft(audio, frame_length=ref_fl, frame_step=ref_fs, fft_length=ref_fl, window_fn=ref_wfn)
    ref_mag = tf.abs(ref_stft)
    ref_T = tf.shape(ref_mag)[0]
    ref_F = tf.shape(ref_mag)[1]

    channels = []
    for i, (fl, fs) in enumerate(configs):
        wfn = _make_window_fn(fl)
        s = tf.signal.stft(audio, frame_length=fl, frame_step=fs, fft_length=fl, window_fn=wfn)
        mag = tf.abs(s)
        log_mag = tf.math.log1p(mag)

        if i == 0:
            channels.append(log_mag)
        else:
            log_mag = log_mag[tf.newaxis, ..., tf.newaxis]
            log_mag = tf.image.resize(log_mag, [ref_T, ref_F])
            channels.append(log_mag[0, ..., 0])

    feat = tf.stack(channels, axis=-1)
    feat = pad_feature(feat)
    return feat


def pad_batch(x):
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]

    pad_h = (CONV_SIZE - h % CONV_SIZE) % CONV_SIZE
    pad_w = (CONV_SIZE - w % CONV_SIZE) % CONV_SIZE

    return tf.pad(
        x,
        paddings=[
            [0, 0],      # batch
            [0, pad_h],  # height (time frames)
            [0, pad_w]   # width (frequency bins)
        ]
    )

# ==========================================================
# Example preprocessing
# ==========================================================

def _random_fir_filter(length=16):
    """Random smooth FIR kernel with unit DC gain — acts as a random EQ."""
    raw = tf.random.normal([length], dtype=tf.float32)
    kernel = tf.cumsum(raw)
    kernel = kernel / (tf.reduce_sum(kernel) + 1e-6)
    return kernel


def _augment_stems(example):
    """Demucs-style augmentation: random gain + random EQ per stem,
    then rebuild the mix from the augmented stems.

    Returns (mix, stems) with stems shaped (4, N).
    """
    stems = tf.stack([
        example["drums"],
        example["bass"],
        example["other"],
        example["vocals"],
    ])

    gains = tf.exp(tf.random.uniform([4, 1], -0.36, 0.36))  # 0.7 .. 1.43 per stem
    stems = stems * gains

    filtered = []
    for i in range(4):
        kernel = _random_fir_filter()[:, tf.newaxis, tf.newaxis]  # (L, 1, 1)
        s = stems[i][tf.newaxis, :, tf.newaxis]                   # (1, N, 1)
        filtered.append(tf.nn.conv1d(s, kernel, stride=1, padding="SAME")[0, :, 0])
    stems = tf.stack(filtered)

    mix = tf.reduce_sum(stems, axis=0)

    gain = tf.random.uniform([], 0.8, 1.25)
    stems = stems * gain
    mix = mix * gain
    return mix, stems


def preprocess(example, configs=None, augment=False):
    if configs is None:
        configs = MR_STFT_CONFIGS

    if augment:
        mix, stems = _augment_stems(example)
        tracks = tf.concat([[mix], stems], axis=0)
    else:
        mix = example["mix"]
        tracks = tf.stack([
            example["mix"],
            example["drums"],
            example["bass"],
            example["other"],
            example["vocals"],
        ])

    # Reference STFT (first config) — used for mask computation
    ref_fl, ref_fs = configs[0]
    ref_wfn = _make_window_fn(ref_fl)
    ref_stft = tf.signal.stft(tracks, frame_length=ref_fl, frame_step=ref_fs, fft_length=ref_fl, window_fn=ref_wfn)
    ref_mag = tf.abs(ref_stft)
    ref_T = tf.shape(ref_mag)[1]
    ref_F = tf.shape(ref_mag)[2]

    # Multi-resolution STFT for mix → input channels
    mix_channels = []
    for i, (fl, fs) in enumerate(configs):
        wfn = _make_window_fn(fl)
        s = tf.signal.stft(mix, frame_length=fl, frame_step=fs, fft_length=fl, window_fn=wfn)
        mag = tf.abs(s)
        log_mag = tf.math.log1p(mag)

        if i == 0:
            mix_channels.append(log_mag)
        else:
            log_mag = log_mag[tf.newaxis, ..., tf.newaxis]
            log_mag = tf.image.resize(log_mag, [ref_T, ref_F])
            mix_channels.append(log_mag[0, ..., 0])

    mix_input = tf.stack(mix_channels, axis=-1)

    # Masks from reference STFT
    masks = ref_mag[1:] / (ref_mag[0] + EPS)
    masks = tf.clip_by_value(masks, 0.0, 1.0)
    masks = tf.transpose(masks, [1, 2, 0])

    source_stft = ref_stft[1:]
    source_real = tf.transpose(tf.math.real(source_stft), [1, 2, 0])
    source_imag = tf.transpose(tf.math.imag(source_stft), [1, 2, 0])

    mix_real = tf.math.real(ref_stft[0])[..., tf.newaxis]
    mix_imag = tf.math.imag(ref_stft[0])[..., tf.newaxis]

    # Pad to CONV_SIZE alignment
    mix_input = pad_feature(mix_input)
    masks = pad_feature(masks)
    source_real = pad_feature(source_real)
    source_imag = pad_feature(source_imag)
    mix_real = pad_feature(mix_real)
    mix_imag = pad_feature(mix_imag)

    target = tf.concat([masks, source_real, source_imag, mix_real, mix_imag], axis=-1)

    return mix_input, target


# ==========================================================
# Statistics
# ==========================================================

class Normalizer:
    def __init__(self, stats_file=None):
        if stats_file is None:
            self.mean = None
            self.std = None

        else:
            stats = np.load(stats_file)
            mean = stats["mean"]
            std = stats["std"]
            # Store plain values: converting inside __call__ keeps the
            # tensors as graph constants. Eager tensors captured by the
            # dataset map would be turned into hidden resource variables on
            # CPU:0, crashing GPU training ("Trying to access resource ...").
            if mean.ndim == 0:
                self.mean = float(mean)
                self.std = float(std)
            else:
                self.mean = np.asarray(mean, dtype=np.float32)
                self.std = np.asarray(std, dtype=np.float32)

    def __call__(self, x, y):
        if self.mean is None:
            return x, y

        mean = tf.constant(self.mean, dtype=tf.float32)
        std = tf.constant(self.std, dtype=tf.float32)

        x = (x - mean) / (std + 1e-8)

        return x, y


def create_dataset(
    tfrecord_dir,
    batch_size,
    stats_file=None,
    shuffle=True,
    shuffle_buffer=128,
    compression_type="GZIP",
    num_parallel=1,
    cycle_length=1,
    prefetch_buffer=1,
    stft_configs=None,
    augment=False,
):
    files=tf.data.Dataset.list_files(
        os.path.join(tfrecord_dir,"*.tfrecord"),
        shuffle=shuffle
    )

    dataset=files.interleave(
        lambda x: tf.data.TFRecordDataset(
            x,
            compression_type=compression_type
        ),
        cycle_length=cycle_length,
        block_length=1,
        num_parallel_calls=num_parallel,
        deterministic=not shuffle
    )

    dataset=dataset.map(parse_tfrecord, num_parallel_calls=num_parallel)

    dataset=dataset.map(
        partial(preprocess, configs=stft_configs, augment=augment),
        num_parallel_calls=num_parallel,
    )
    normalizer=Normalizer(stats_file)
    dataset=dataset.map(normalizer, num_parallel_calls=num_parallel)
    dataset=dataset.batch(batch_size, drop_remainder=False)
    dataset=dataset.prefetch(prefetch_buffer)

    return dataset


if __name__=="__main__":
    train=create_dataset(
        "../musdb18/tfrecord",
        batch_size=4,
        stats_file=None,
        shuffle=True
    )

    for x,y in train.take(1):
        print("Input :",x.shape)
        print("Target:",y.shape)
        print()

        print("Input dtype:", x.dtype)
        print("Target dtype:", y.dtype)
        print()

        print(
            "Input range:",
            tf.reduce_min(x).numpy(),
            tf.reduce_max(x).numpy()
        )
        print()

        print("Mask range:", tf.reduce_min(y).numpy(), tf.reduce_max(y).numpy())