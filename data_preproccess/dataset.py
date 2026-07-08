import os
import numpy as np
import tensorflow as tf

# ==========================================================
# CONFIG
# ==========================================================

FRAME_LENGTH = 2048
FRAME_STEP = 512
CONV_SIZE = 16
EPS = 1e-4

SQRT_HANN_WINDOW = tf.sqrt(
    tf.signal.hann_window(
        FRAME_LENGTH,
        periodic=True
    )
)


# ==========================================================
# STFT
# ==========================================================

def sqrt_hann_window(frame_length, dtype):
    return tf.cast(SQRT_HANN_WINDOW, dtype)


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
# STFT
# ==========================================================

def waveform_to_stft(audio):
    return tf.signal.stft(
        audio,
        frame_length=FRAME_LENGTH,
        frame_step=FRAME_STEP,
        fft_length=FRAME_LENGTH,
        window_fn=sqrt_hann_window
    )


# ==========================================================
# Padding
# ==========================================================

def pad_feature(x):
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]

    pad_h = (CONV_SIZE - h % CONV_SIZE) % CONV_SIZE
    pad_w = (CONV_SIZE - w % CONV_SIZE) % CONV_SIZE

    x = tf.pad(
        x,
        [
            [0, pad_h],
            [0, pad_w]
        ]
    )

    return x


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

def waveform_to_features(audio):
    stft = waveform_to_stft(audio)

    mag = magnitude(stft)

    log_mag = log_magnitude(mag)

    mag = pad_feature(mag)
    log_mag = pad_feature(log_mag)

    mag = mag[..., tf.newaxis]
    log_mag = log_mag[..., tf.newaxis]

    return mag, log_mag


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

def preprocess(example):
    tracks = tf.stack([
        example["mix"],
        example["drums"],
        example["bass"],
        example["other"],
        example["vocals"]
    ])

    stft = waveform_to_stft(tracks)

    mag = tf.abs(stft)

    log_mag = tf.math.log1p(mag)

    mag = pad_batch(mag)
    log_mag = pad_batch(log_mag)

    mix = log_mag[0][...,None]

    masks = mag[1:] / (mag[0] + EPS)
    masks = tf.clip_by_value(masks, 0.0, 1.0)

    masks = tf.transpose(
        masks,
        [1,2,0]
    )

    return mix, masks


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

            self.mean = tf.constant(stats["mean"], dtype=tf.float32)
            self.std = tf.constant(stats["std"], dtype=tf.float32)

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

    dataset=dataset.map(preprocess, num_parallel_calls=num_parallel)
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