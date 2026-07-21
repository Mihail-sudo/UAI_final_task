import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data_preproccess"))

import numpy as np
import tensorflow as tf

from data_preproccess.dataset import (
    parse_tfrecord,
    MR_STFT_CONFIGS,
    _make_window_fn,
    pad_feature,
    CONV_SIZE,
    EPS,
)
from refine import create_refine_model


TFRECORD_DIR = "musdb18/tfrecord"
BATCH_SIZE = 1
EPOCHS = 50
LEARNING_RATE = 1e-4

ref_fl, ref_fs = MR_STFT_CONFIGS[0]
ref_wfn = _make_window_fn(ref_fl)


def build_mr_input(mix_wave):
    ref_stft = tf.signal.stft(
        mix_wave, frame_length=ref_fl, frame_step=ref_fs,
        fft_length=ref_fl, window_fn=ref_wfn,
    )
    ref_mag = tf.abs(ref_stft)
    ref_T = tf.shape(ref_mag)[0]
    ref_F = tf.shape(ref_mag)[1]

    channels = []
    for i, (fl, fs) in enumerate(MR_STFT_CONFIGS):
        wfn = _make_window_fn(fl)
        s = tf.signal.stft(
            mix_wave, frame_length=fl, frame_step=fs,
            fft_length=fl, window_fn=wfn,
        )
        log_mag = tf.math.log1p(tf.abs(s))
        if i == 0:
            channels.append(log_mag)
        else:
            log_mag = tf.image.resize(
                log_mag[tf.newaxis, :, :, tf.newaxis], [ref_T, ref_F]
            )
            channels.append(log_mag[0, :, :, 0])

    feat = tf.stack(channels, axis=-1)
    feat = pad_feature(feat)
    return feat, ref_stft


def separate_sources(mix_wave, separator):
    feat, ref_stft = build_mr_input(mix_wave)
    feat = feat[tf.newaxis, :, :, :]
    pred = separator.predict(feat, verbose=0)[0]

    mask_r = pred[:, :, :4]
    mask_i = pred[:, :, 4:]

    ref_real = tf.math.real(ref_stft)[..., tf.newaxis]
    ref_imag = tf.math.imag(ref_stft)[..., tf.newaxis]

    src_r = mask_r * ref_real - mask_i * ref_imag
    src_i = mask_r * ref_imag + mask_i * ref_real

    T_orig = tf.shape(mix_wave)[0]
    sources = []
    for i in range(4):
        source_stft = tf.complex(src_r[:, :, i], src_i[:, :, i])
        source = tf.signal.inverse_stft(
            source_stft,
            frame_length=ref_fl,
            frame_step=ref_fs,
            fft_length=ref_fl,
            window_fn=ref_wfn,
        )
        source = source[:T_orig]
        sources.append(source)

    return tf.stack(sources, axis=-1)


def get_raw_dataset(tfrecord_dir, shuffle=True):
    files = tf.data.Dataset.list_files(
        os.path.join(tfrecord_dir, "*.tfrecord"), shuffle=shuffle
    )
    ds = files.interleave(
        lambda x: tf.data.TFRecordDataset(x, compression_type="GZIP"),
        cycle_length=2,
        block_length=1,
        num_parallel_calls=2,
        deterministic=not shuffle,
    )
    if shuffle:
        ds = ds.shuffle(128, reshuffle_each_iteration=True)
    ds = ds.map(parse_tfrecord, num_parallel_calls=2)
    ds = ds.batch(1)
    ds = ds.prefetch(1)
    return ds


class RefineTrainer:
    def __init__(self, separator_path="separator.keras"):
        self.separator = tf.keras.models.load_model(separator_path)
        self.separator.trainable = False

        self.refine = create_refine_model()
        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    def train_step(self, batch):
        mix = batch["mix"][0]
        targets = tf.stack(
            [batch["drums"][0], batch["bass"][0], batch["other"][0], batch["vocals"][0]],
            axis=-1,
        )

        with tf.GradientTape() as tape:
            separated = separate_sources(mix, self.separator)
            separated = tf.reshape(separated, [-1, 1])
            targets_flat = tf.reshape(targets, [-1, 1])

            refined = self.refine(separated[tf.newaxis, :, :])[0]
            loss = tf.reduce_mean(tf.abs(refined - targets_flat))

        grads = tape.gradient(loss, self.refine.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)
        self.optimizer.apply_gradients(zip(grads, self.refine.trainable_variables))

        return loss

    def train(self, epochs=EPOCHS):
        train_ds = get_raw_dataset(TFRECORD_DIR)
        n_batches = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_steps = 0

            for batch in train_ds:
                loss = self.train_step(batch)
                epoch_loss += loss.numpy()
                n_steps += 1
                n_batches += 1

                if n_steps % 50 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, step {n_steps}, loss={epoch_loss/n_steps:.6f}")

            avg_loss = epoch_loss / max(n_steps, 1)
            print(f"Epoch {epoch+1}/{epochs} done, avg loss={avg_loss:.6f}")

            self.refine.save_weights(f"refine_ep{epoch+1:02d}.weights.h5")

        self.refine.save("refine.keras")
        print("Saved refine.keras")


if __name__ == "__main__":
    trainer = RefineTrainer()
    trainer.train()
