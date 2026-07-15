import tensorflow as tf


def _sqrt_hann_window(frame_length):
    window = tf.sqrt(tf.signal.hann_window(frame_length, periodic=True))
    def _fn(fl, dtype):
        return tf.cast(window, dtype)
    return _fn


class SeparatorLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        gamma=0.1,
        scales=(1, 2, 4),
        ref_frame_length=4096,
        ref_frame_step=1024,
        name="separator_loss",
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.scales = scales
        self.ref_fl = ref_frame_length
        self.ref_fs = ref_frame_step
        self.ref_wfn = _sqrt_hann_window(ref_frame_length)

    def call(self, y_true, y_pred):
        target_masks = y_true[..., :4]
        mix_mag = y_true[..., 4:5]
        mix_phase = y_true[..., 5:6]

        mask_loss = tf.reduce_mean(tf.abs(target_masks - y_pred))

        spec_loss = 0.0
        for s in self.scales:
            pred_src = mix_mag * y_pred
            target_src = mix_mag * target_masks
            if s > 1:
                pred_src = tf.nn.avg_pool2d(pred_src, s, s, "SAME")
                target_src = tf.nn.avg_pool2d(target_src, s, s, "SAME")
            spec_loss += tf.reduce_mean(tf.abs(pred_src - target_src))
        spec_loss /= len(self.scales)

        return self.alpha * mask_loss + self.beta * spec_loss


CombinedSpectrogramLoss = SeparatorLoss
