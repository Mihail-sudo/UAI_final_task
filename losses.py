import tensorflow as tf


EPS = 1e-8


def mask_l1(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def mask_bce(y_true, y_pred):
    return tf.reduce_mean(
        tf.keras.losses.binary_crossentropy(y_true, y_pred)
    )


def mask_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def spectrogram_l1(mix_mag, pred_masks, target_masks):
    pred_spec = pred_masks * mix_mag
    target_spec = target_masks * mix_mag
    return tf.reduce_mean(tf.abs(pred_spec - target_spec))


def combined_loss(mix_mag, pred_masks, target_masks, alpha=1.0, beta=1.0):
    bce = mask_bce(target_masks, pred_masks)
    spec_l1 = spectrogram_l1(mix_mag, pred_masks, target_masks)
    return alpha * bce + beta * spec_l1


class CombinedLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=1.0, name="combined_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        return self.alpha * mask_bce(y_true, y_pred) + self.beta * mask_l1(y_true, y_pred)


class CombinedSpectrogramLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, beta=1e-3, name="combined_spectrogram_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        target_masks = y_true[..., :4]
        mix_mag = y_true[..., 4:5]
        bce = mask_bce(target_masks, y_pred)
        spec_l1 = spectrogram_l1(mix_mag, y_pred, target_masks)
        return self.alpha * bce + self.beta * spec_l1
