import tensorflow as tf


def mask_bce(y_true, y_pred):
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )


def mask_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true[..., :4] - y_pred))


class MultiResolutionSpectrogramLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        alpha=1.0,
        beta=1e-3,
        scales=(1, 2, 4),
        name="multi_res_spec_loss",
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.scales = scales

    def call(self, y_true, y_pred):
        target_masks = y_true[..., :4]
        mix_mag = y_true[..., 4:5]

        bce = mask_bce(target_masks, y_pred)
        spec_loss = 0.0

        for s in self.scales:
            pred_src = mix_mag * y_pred
            target_src = mix_mag * target_masks

            if s > 1:
                pred_src = tf.nn.avg_pool2d(pred_src, s, s, "SAME")
                target_src = tf.nn.avg_pool2d(target_src, s, s, "SAME")

            spec_loss += tf.reduce_mean(tf.abs(pred_src - target_src))

        spec_loss /= len(self.scales)
        return self.alpha * bce + self.beta * spec_loss


CombinedSpectrogramLoss = MultiResolutionSpectrogramLoss
