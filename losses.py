import tensorflow as tf


EPS = 1e-8


def _weighted_mean(x, w):
    if w is None:
        return tf.reduce_mean(x)
    return tf.reduce_sum(x * w) / (tf.reduce_sum(w) + EPS)


class ComplexSeparatorLoss(tf.keras.losses.Loss):
    """Loss for the complex-mask separator.

    The model outputs a complex mask (Re, Im) per source; sources are decoded as
    ``mask * mix_stft``. The loss is computed on that reconstruction only:
    a phase-aware complex STFT loss plus a multi-scale spectral loss.

    The legacy magnitude-ratio mask target (|S| / |mix|, channels [0:4]) is
    disabled by default (``alpha=0``): it contradicts the phase-aware
    reconstruction in bins where |S| > |mix| (phase cancellation, ~15-25% of
    bins) and is arbitrary noise in noise-floor bins.

    Per-pixel weighting by the mix magnitude (``mix_weight=True``) downweights
    noise-floor bins smoothly — no frequency bins are cut.
    """

    def __init__(
        self,
        alpha=0.0,
        beta=1.0,
        gamma=0.1,
        scales=(1, 2, 4),
        mix_weight=True,
        weight_tau=0.05,
        name="complex_separator_loss",
    ):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.scales = scales
        self.mix_weight = mix_weight
        self.weight_tau = weight_tau

    def call(self, y_true, y_pred):
        target_mag_masks = y_true[..., :4]
        tgt_r = y_true[..., 4:8]
        tgt_i = y_true[..., 8:12]
        mix_r = y_true[..., 12:13]
        mix_i = y_true[..., 13:14]

        pred_r = y_pred[..., :4]
        pred_i = y_pred[..., 4:8]

        # Per-pixel weight: ~1 where the mix has energy, small floor in the
        # noise floor. Full frequency range is kept.
        w = None
        if self.mix_weight:
            mix_mag = tf.sqrt(mix_r ** 2 + mix_i ** 2 + EPS)
            w = tf.minimum(mix_mag / self.weight_tau, 1.0)
            w = tf.maximum(w, 0.01)

        # Optional magnitude-ratio mask target (off by default — see docstring).
        pred_mag = tf.sqrt(pred_r ** 2 + pred_i ** 2 + EPS)
        mag_loss = _weighted_mean(tf.abs(pred_mag - target_mag_masks), w)

        # Phase-aware reconstruction: mask * mix_stft vs source stft.
        src_r = pred_r * mix_r - pred_i * mix_i
        src_i = pred_r * mix_i + pred_i * mix_r
        stft_mag = tf.sqrt((src_r - tgt_r) ** 2 + (src_i - tgt_i) ** 2 + EPS)
        stft_loss = _weighted_mean(stft_mag, w)

        # Multi-scale spectral loss on reconstructed source magnitudes.
        spec_loss = 0.0
        for s in self.scales:
            pred_src_mag = tf.sqrt(src_r ** 2 + src_i ** 2 + EPS)
            tgt_src_mag = tf.sqrt(tgt_r ** 2 + tgt_i ** 2 + EPS)
            w_s = w
            if s > 1:
                pred_src_mag = tf.nn.avg_pool2d(pred_src_mag, s, s, "SAME")
                tgt_src_mag = tf.nn.avg_pool2d(tgt_src_mag, s, s, "SAME")
                if w is not None:
                    w_s = tf.nn.avg_pool2d(w, s, s, "SAME")
            spec_loss += _weighted_mean(tf.abs(pred_src_mag - tgt_src_mag), w_s)
        spec_loss /= len(self.scales)

        return self.alpha * mag_loss + self.beta * stft_loss + self.gamma * spec_loss


CombinedSpectrogramLoss = ComplexSeparatorLoss
