import numpy as np
from dataset import create_dataset


def compute_stats(tfrecord_dir, output_file="stats.npz", batch_size=8):
    dataset = create_dataset(
        tfrecord_dir=tfrecord_dir,
        batch_size=batch_size,
        stats_file=None,
        shuffle=False
    )

    print("Computing per-channel statistics...")

    n_channels = None
    pixel_sum = None
    pixel_sq_sum = None
    pixel_count = 0

    for step, (x, _) in enumerate(dataset):
        x = x.numpy().astype(np.float64)

        if n_channels is None:
            n_channels = x.shape[-1]
            pixel_sum = np.zeros(n_channels, dtype=np.float64)
            pixel_sq_sum = np.zeros(n_channels, dtype=np.float64)

        pixel_sum += x.sum(axis=(0, 1, 2))
        pixel_sq_sum += (x ** 2).sum(axis=(0, 1, 2))
        pixel_count += x.shape[0] * x.shape[1] * x.shape[2]

        if (step + 1) % 100 == 0:
            print(f"Processed batches: {step + 1}  (channels: {n_channels})")

    mean = pixel_sum / pixel_count
    var = pixel_sq_sum / pixel_count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-12))

    np.savez(output_file, mean=mean.astype(np.float32), std=std.astype(np.float32))

    print()
    print("Done")
    for i in range(n_channels):
        print(f"  Channel {i}: mean={mean[i]:.6f}  std={std[i]:.6f}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    compute_stats(
        tfrecord_dir="musdb18/tfrecord",
        output_file="musdb18/stats.npz",
        batch_size=8
    )