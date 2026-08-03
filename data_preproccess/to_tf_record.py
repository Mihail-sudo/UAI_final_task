import os
import io
import random

import ffmpeg
import numpy as np
import soundfile as sf
import tensorflow as tf

# ==========================================================
# CONFIG
# ==========================================================

GLOBAL_RATE = 44100

CHUNK_SEC = 4
NUM_CHUNKS = 25
CHUNK_SAMPLES = GLOBAL_RATE * CHUNK_SEC

OUTPUT_SAMPLES_PER_FILE = 50

# ==========================================================
# AUDIO
# ==========================================================


def extract_audio_to_memory(input_file, track_index):
    process = (
        ffmpeg
        .input(input_file)
        .output(
            "pipe:",
            format="wav",
            acodec="pcm_s16le",
            ar=GLOBAL_RATE,
            ac=1,
            map=track_index
        )
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    out, err = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(err.decode())

    return out


def load_audio_from_memory(audio_bytes):
    audio_buffer = io.BytesIO(audio_bytes)
    waveform, sr = sf.read(audio_buffer, dtype="float32")

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    return waveform.astype(np.float32), sr


def load_audio(input_file, track):
    audio_bytes = extract_audio_to_memory(input_file, track)
    return load_audio_from_memory(audio_bytes)


# ==========================================================
# MUSDB
# ==========================================================

def load_song(path):

    mix, _ = load_audio(path, "0:a:0")
    drums, _ = load_audio(path, "0:a:1")
    bass, _ = load_audio(path, "0:a:2")
    other, _ = load_audio(path, "0:a:3")
    vocals, _ = load_audio(path, "0:a:4")

    return mix, drums, bass, other, vocals


# ==========================================================
# CHUNK
# ==========================================================

def random_chunk(*tracks):
    length = len(tracks[0])
    if length <= CHUNK_SAMPLES:
        result = []
        for track in tracks:
            pad = CHUNK_SAMPLES - len(track)
            result.append(
                np.pad(track, (0, pad), mode="constant")
            )
        return result

    start = random.randint(0, length - CHUNK_SAMPLES)

    end = start + CHUNK_SAMPLES

    return [
        track[start:end]
        for track in tracks
    ]


# ==========================================================
# TFRecord helpers
# ==========================================================

def _bytes_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(
            value=[value]
        )
    )


def _int_feature(value):
    return tf.train.Feature(
        int64_list=tf.train.Int64List(
            value=[value]
        )
    )


def serialize_example(mix, drums, bass, other, vocals):
    feature = {
        "length": _int_feature(len(mix)),
        "mix": _bytes_feature(mix.astype(np.float32).tobytes()),
        "drums": _bytes_feature(drums.astype(np.float32).tobytes()),
        "bass": _bytes_feature(bass.astype(np.float32).tobytes()),
        "other": _bytes_feature(other.astype(np.float32).tobytes()),
        "vocals": _bytes_feature(vocals.astype(np.float32).tobytes())
    }

    example = tf.train.Example(
        features=tf.train.Features(
            feature=feature
        )
    )

    return example.SerializeToString()


# ==========================================================
# WRITER
# ==========================================================

class TFRecordWriter:
    def __init__(self, output_dir, prefix="train", samples_per_file=100):

        self.output_dir = output_dir
        self.prefix = prefix
        self.samples_per_file = samples_per_file

        os.makedirs(output_dir, exist_ok=True)

        self.file_index = 0
        self.counter = 0

        self.writer = None

        self._open_new_file()


    def _open_new_file(self):
        if self.writer is not None:
            self.writer.close()

        filename = os.path.join(
            self.output_dir,
            f"{self.prefix}-{self.file_index:03d}.tfrecord"
        )

        options = tf.io.TFRecordOptions(compression_type="GZIP")

        self.writer = tf.io.TFRecordWriter(filename, options=options)

        print("Create:", filename)

        self.file_index += 1
        self.counter = 0


    def write(self, mix, drums, bass, other, vocals):
        example = serialize_example(mix, drums, bass, other, vocals)

        self.writer.write(example)
        self.counter += 1

        if self.counter >= self.samples_per_file:
            self._open_new_file()

    def close(self):
        if self.writer is not None:
            self.writer.close()


# ==========================================================
# MAIN
# ==========================================================


def preprocess_dataset(input_dir, output_dir, num_chunks=NUM_CHUNKS):
    files = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
    )

    writer = TFRecordWriter(
        output_dir,
        prefix="train",
        samples_per_file=OUTPUT_SAMPLES_PER_FILE
    )

    total = 0

    for file in files:
        print(file)
        mix, drums, bass, other, vocals = load_song(file)

        for _ in range(num_chunks):
            mix_chunk, drums_chunk, bass_chunk, other_chunk, vocals_chunk = random_chunk(mix, drums, bass, other, vocals)

            writer.write(
                mix_chunk,
                drums_chunk,
                bass_chunk,
                other_chunk,
                vocals_chunk
            )
            total += 1

    writer.close()

    print()
    print("Done.")
    print("Examples:", total)
    print("TFRecord files:", writer.file_index)
    print("TFRecord files:", writer.file_index)


# ==========================================================
# ENTRY
# ==========================================================

if __name__ == "__main__":
    preprocess_dataset(
        input_dir="musdb18/wav/train",
        output_dir="musdb18/tfrecord"
    )