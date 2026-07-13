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
    (2048, 512),   # reference: high frequency resolution
    (1024, 256),   # high time resolution
]
CONV_SIZE = 16
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

def preprocess(example, configs=None):
    if configs is None:
        configs = MR_STFT_CONFIGS

    tracks = tf.stack([
        example["mix"],
        example["drums"],
        example["bass"],
        example["other"],
        example["vocals"],
    ])

    mix = example["mix"]

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
    mix_mag_ref = ref_mag[0]
    masks = ref_mag[1:] / (ref_mag[0] + EPS)
    masks = tf.clip_by_value(masks, 0.0, 1.0)
    masks = tf.transpose(masks, [1, 2, 0])

    # Pad to CONV_SIZE alignment
    mix_input = pad_feature(mix_input)
    masks = pad_feature(masks)
    mix_mag_ref = pad_feature(mix_mag_ref)

    target = tf.concat([masks, mix_mag_ref[..., tf.newaxis]], axis=-1)

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

            if mean.ndim == 0:
                self.mean = tf.constant(float(mean), dtype=tf.float32)
                self.std = tf.constant(float(std), dtype=tf.float32)
            else:
                self.mean = tf.constant(mean, dtype=tf.float32)
                self.std = tf.constant(std, dtype=tf.float32)

    def __call__(self, x, y):
        if self.mean is None:
            return x, y

        x = (x - self.mean) / (self.std + 1e-8)

        return x, y


def create_dataset(
    tfrecord_dir,
    batch_size,
    stats_file=None,
    shuffle=True,
    shuffle_buffer=128,
    compression_type="GZIP",
    num_parallel=2,
    cycle_length=2,
    prefetch_buffer=2,
    stft_configs=None,
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

    if shuffle:
        dataset=dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    dataset=dataset.map(parse_tfrecord, num_parallel_calls=num_parallel)

    dataset=dataset.map(
        partial(preprocess, configs=stft_configs),
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