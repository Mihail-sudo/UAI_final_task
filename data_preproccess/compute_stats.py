import numpy as np
from dataset import create_dataset


def compute_stats(tfrecord_dir, output_file="stats.npz", batch_size=8):
    dataset = create_dataset(
        tfrecord_dir=tfrecord_dir,
        batch_size=batch_size,
        stats_file=None,
        shuffle=False
    )

    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    pixel_count = 0

    print("Computing statistics...")

    for step, (x, _) in enumerate(dataset):

        x = x.numpy()

        pixel_sum += x.sum(dtype=np.float64)
        pixel_sq_sum += np.square(x, dtype=np.float64).sum(dtype=np.float64)
        pixel_count += x.size

        if (step + 1) % 100 == 0:
            print(f"Processed batches: {step + 1}")

    mean = pixel_sum / pixel_count
    var = pixel_sq_sum / pixel_count - mean ** 2
    std = np.sqrt(max(var, 1e-12))

    np.savez(output_file, mean=np.float32(mean), std=np.float32(std))

    print()
    print("Done")
    print(f"Mean : {mean:.6f}")
    print(f"Std  : {std:.6f}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    compute_stats(
        tfrecord_dir="../musdb18/tfrecord",
        output_file="../musdb18/stats.npz",
        batch_size=8
    )