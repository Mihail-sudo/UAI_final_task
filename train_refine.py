"""Training script for the waveform refinement model.

Uses the frozen separator (``separator.keras``) to generate (input, target) pairs:
    input  = separated stem waveform (OLA + complex masking, see inference.separate)
    target = the true stem waveform

Pairs are drawn from the same TFRecord dataset the separator was trained on
(``musdb18/tfrecord``, GZIP, 4s chunks): each record already contains the raw
mix + 4 true stems, so no ffmpeg decoding is needed inside the training loop.

Training runs as a manual GradientTape loop: a generator that iterates a tf.data
pipeline and runs eager separation cannot be used through ``Dataset.from_generator
+ model.fit`` (nested tf.data iteration inside a generator deadlocks fit()).

The refinement model (refine.py) is trained with SI-SDR loss.
"""

import argparse
import glob
import os
import random
import time

import numpy as np
import tensorflow as tf

from data_preproccess.dataset import MR_STFT_CONFIGS, parse_tfrecord
from inference import separate, si_sdr
from refine import create_refine_model

STEM_NAMES = ["drums", "bass", "other", "vocals"]


# ==========================================================
# SI-SDR loss
# ==========================================================

def si_sdr_loss(target, estimate):
    target = target - tf.reduce_mean(target, axis=-1, keepdims=True)
    estimate = estimate - tf.reduce_mean(estimate, axis=-1, keepdims=True)
    alpha = tf.reduce_sum(estimate * target, axis=-1, keepdims=True) / (
        tf.reduce_sum(target * target, axis=-1, keepdims=True) + 1e-8
    )
    scaled = alpha * target
    noise = estimate - scaled
    sdr = 10.0 * tf.math.log(
        tf.reduce_sum(scaled ** 2, axis=-1)
        / (tf.reduce_sum(noise ** 2, axis=-1) + 1e-8)
        + 1e-8
    ) / tf.math.log(10.0)
    return -sdr


# ==========================================================
# Pair generation (from TFRecords, no ffmpeg in the loop)
# ==========================================================

def pair_generator(files, separator_model, configs, stats_file, shuffle_buffer=64):
    """Infinite stream of (separated_stem, true_stem) waveform pairs."""
    ds = tf.data.Dataset.list_files(files, shuffle=True)
    ds = ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, compression_type="GZIP"),
        cycle_length=len(files),
        block_length=1,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    for raw in ds:
        example = parse_tfrecord(raw)
        mix = example["mix"].numpy()
        true = {name: example[name].numpy() for name in STEM_NAMES}

        separated = separate(mix, separator_model, configs=configs, stats_file=stats_file)

        for name in STEM_NAMES:
            yield (
                separated[..., STEM_NAMES.index(name)].astype(np.float32)[..., None],
                true[name].astype(np.float32)[..., None],
            )


def next_batch(gen, batch_size):
    xs, ys = [], []
    for _ in range(batch_size):
        x, y = next(gen)
        xs.append(x)
        ys.append(y)
    return (
        np.stack(xs, axis=0).astype(np.float32),
        np.stack(ys, axis=0).astype(np.float32),
    )


# ==========================================================
# Training (manual GradientTape loop)
# ==========================================================

def train(refine_model, files, separator_model, configs, stats_file,
          batch_size=8, steps=2000, lr=1e-3, log_every=100):
    gen = pair_generator(files, separator_model, configs, stats_file)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_mean = tf.keras.metrics.Mean()

    t0 = time.time()
    for step in range(1, steps + 1):
        xb, yb = next_batch(gen, batch_size)

        with tf.GradientTape() as tape:
            pred = refine_model(xb, training=True)
            # separate() outputs 176128 samples (ISTFT length) while the
            # target is the full 176400-sample chunk — crop to match
            yb_crop = yb[:, : pred.shape[1]]
            loss = tf.reduce_mean(si_sdr_loss(yb_crop, pred))

        grads = tape.gradient(loss, refine_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, refine_model.trainable_variables))

        loss_mean.update_state(loss)

        if step % log_every == 0:
            elapsed = time.time() - t0
            print(f"step {step:5d}/{steps}  loss={loss_mean.result():.4f} "
                  f"(SDR {(-float(loss_mean.result())):.2f} dB)  {elapsed:.0f}s")
            loss_mean.reset_state()

    return refine_model


# ==========================================================
# Quick evaluation: SI-SDR before / after refinement
# ==========================================================

def evaluate(files, separator_model, refine_model, stats_file, n_chunks=2):
    sdrs_before, sdrs_after = [], []
    for _ in range(n_chunks):
        file = random.choice(files)
        ds = tf.data.TFRecordDataset(file, compression_type="GZIP")
        example = parse_tfrecord(next(iter(ds.shuffle(64))))
        mix = example["mix"].numpy()
        true = {name: example[name].numpy() for name in STEM_NAMES}

        before = separate(mix, separator_model, configs=MR_STFT_CONFIGS, stats_file=stats_file)
        after = separate(mix, separator_model, configs=MR_STFT_CONFIGS,
                         stats_file=stats_file, refine_model=refine_model)
        for name in STEM_NAMES:
            i = STEM_NAMES.index(name)
            T = min(before.shape[0], len(true[name]))
            sdrs_before.append(si_sdr(before[:T, i], true[name][:T]))
            sdrs_after.append(si_sdr(after[:T, i], true[name][:T]))
    print(f"SI-SDR before refine: {np.mean(sdrs_before):.2f} dB")
    print(f"SI-SDR after  refine: {np.mean(sdrs_after):.2f} dB")


# ==========================================================
# MAIN
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--separator", default="separator.keras")
    parser.add_argument("--output", default="refine.keras")
    parser.add_argument("--tfrecord-dir", default="musdb18/tfrecord")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stats", default="musdb18/stats.npz")
    parser.add_argument("--no-stats", action="store_true", help="Disable input normalization")
    args = parser.parse_args()

    if not os.path.exists(args.separator):
        print("Separator checkpoint not found:", args.separator)
        return

    stats_file = None if args.no_stats else args.stats
    if stats_file is not None and not os.path.exists(stats_file):
        print("Stats file not found, disabling normalization")
        stats_file = None

    files = sorted(glob.glob(os.path.join(args.tfrecord_dir, "train-*.tfrecord")))
    if not files:
        print("No tfrecords found in:", args.tfrecord_dir)
        return
    print("TFRecord files:", len(files))

    separator_model = tf.keras.models.load_model(args.separator)
    refine_model = create_refine_model()

    train(
        refine_model, files, separator_model, MR_STFT_CONFIGS, stats_file,
        batch_size=args.batch_size, steps=args.steps, lr=args.lr,
    )

    evaluate(files, separator_model, refine_model, stats_file)

    refine_model.save(args.output)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
