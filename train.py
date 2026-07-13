import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "data_preproccess"))

import tensorflow as tf
from data_preproccess.dataset import create_dataset, MR_STFT_CONFIGS
from model import create_unet
from losses import CombinedSpectrogramLoss


def mask_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[..., :4] - y_pred))

TFRECORD_DIR = "musdb18/tfrecord"
TEST_TFRECORD_DIR = "musdb18/tfrecord_test"
STATS_FILE = "musdb18/stats.npz"
BATCH_SIZE = 4
EPOCHS = 100
LEARNING_RATE = 1e-4

def main():
    if not os.path.isdir(TFRECORD_DIR) or not any(
        f.endswith(".tfrecord") for f in os.listdir(TFRECORD_DIR)
    ):
        print(
            f"No TFRecord files found at {TFRECORD_DIR}.\n"
            f"Run first: python data_preproccess/to_tf_record.py"
        )
        return

    train_ds = create_dataset(
        tfrecord_dir=TFRECORD_DIR,
        batch_size=BATCH_SIZE,
        stats_file=STATS_FILE if os.path.exists(STATS_FILE) else None,
        shuffle=True,
    )

    val_ds = None
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=5, min_lr=1e-7
        ),
    ]

    if os.path.isdir(TEST_TFRECORD_DIR):
        val_ds = create_dataset(
            tfrecord_dir=TEST_TFRECORD_DIR,
            batch_size=BATCH_SIZE,
            stats_file=STATS_FILE if os.path.exists(STATS_FILE) else None,
            shuffle=False,
        )
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                "checkpoint.keras", save_best_only=True, monitor="val_loss"
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            ),
        ]

    n_channels = len(MR_STFT_CONFIGS)
    model = create_unet(input_shape=(None, None, n_channels))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=CombinedSpectrogramLoss(alpha=1.0, beta=1e-3),
        metrics=[mask_mae],
    )

    model.summary()

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        callbacks=callbacks,
    )

    model.save("separator.keras")


if __name__ == "__main__":
    main()
