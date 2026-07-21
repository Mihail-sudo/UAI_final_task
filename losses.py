import tensorflow as tf


EPS = 1e-8


class ComplexSeparatorLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        alpha=1.0,
        beta=1.0,
        gamma=0.1,
        scales=(1, 2, 4),
        name="complex_separator_loss",
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.scales = scales

    def call(self, y_true, y_pred):
        target_mag_masks = y_true[..., :4]
        tgt_r = y_true[..., 4:8]
        tgt_i = y_true[..., 8:12]
        mix_r = y_true[..., 12:13]
        mix_i = y_true[..., 13:14]

        pred_r = y_pred[..., :4]
        pred_i = y_pred[..., 4:8]

        pred_mag = tf.sqrt(pred_r ** 2 + pred_i ** 2 + EPS)
        mag_loss = tf.reduce_mean(tf.abs(pred_mag - target_mag_masks))

        src_r = pred_r * mix_r - pred_i * mix_i
        src_i = pred_r * mix_i + pred_i * mix_r

        stft_mag = tf.sqrt((src_r - tgt_r) ** 2 + (src_i - tgt_i) ** 2 + EPS)
        stft_loss = tf.reduce_mean(stft_mag)

        spec_loss = 0.0
        for s in self.scales:
            pred_src_mag = tf.sqrt(src_r ** 2 + src_i ** 2 + EPS)
            tgt_src_mag = tf.sqrt(tgt_r ** 2 + tgt_i ** 2 + EPS)
            if s > 1:
                pred_src_mag = tf.nn.avg_pool2d(pred_src_mag, s, s, "SAME")
                tgt_src_mag = tf.nn.avg_pool2d(tgt_src_mag, s, s, "SAME")
            spec_loss += tf.reduce_mean(tf.abs(pred_src_mag - tgt_src_mag))
        spec_loss /= len(self.scales)

        return self.alpha * mag_loss + self.beta * stft_loss + self.gamma * spec_loss


CombinedSpectrogramLoss = ComplexSeparatorLoss
