import tensorflow as tf
from tensorflow.keras import layers, models


def glu(x, n_out):
    a, b = tf.split(x, 2, axis=-1)
    return a * tf.sigmoid(b)


class GLU(layers.Layer):
    def __init__(self, n_out, **kwargs):
        super().__init__(**kwargs)
        self.n_out = n_out

    def call(self, x):
        a, b = tf.split(x, 2, axis=-1)
        return a * tf.sigmoid(b)


class ConvBlock(layers.Layer):
    def __init__(self, in_ch, out_ch, kernel=7, stride=1, dilation=1):
        super().__init__()
        self.conv = layers.Conv1D(
            out_ch * 2, kernel, strides=stride,
            padding="same", dilation_rate=dilation,
            use_bias=False,
        )
        self.norm = layers.GroupNormalization(groups=min(out_ch // 4, 32))
        self.skip = None
        if stride > 1 or in_ch != out_ch:
            self.skip = layers.Conv1D(out_ch, 1, strides=stride, use_bias=False)

    def call(self, x):
        residual = x
        x = self.conv(x)
        x = self.norm(x)
        x = glu(x, x.shape[-1] // 2)
        if self.skip is not None:
            residual = self.skip(residual)
        return x + residual


class TransposeBlock(layers.Layer):
    def __init__(self, in_ch, out_ch, kernel=7, stride=2):
        super().__init__()
        self.conv = layers.Conv1DTranspose(
            out_ch * 2, kernel, strides=stride,
            padding="same", use_bias=False,
        )
        self.norm = layers.GroupNormalization(groups=min(out_ch // 4, 32))
        self.proj = layers.Conv1D(out_ch, 1, use_bias=False)

    def call(self, x, skip):
        x = self.conv(x)
        x = self.norm(x)
        x = glu(x, x.shape[-1] // 2)
        x = x[:, :tf.shape(skip)[1], :]
        x = tf.concat([x, skip], axis=-1)
        x = self.proj(x)
        return x


def create_refine_model(
    n_encoder_levels=5,
    encoder_channels=(8, 16, 32, 64, 96),
    kernel_size=7,
):
    n_levels = min(n_encoder_levels, len(encoder_channels))
    inputs = layers.Input(shape=(None, 1))

    x = inputs
    skips = []

    for i in range(n_levels):
        in_ch = 1 if i == 0 else encoder_channels[i - 1]
        out_ch = encoder_channels[i]
        stride = 2 if i > 0 else 1
        if i == 0:
            x = layers.Conv1D(out_ch, kernel_size, padding="same", use_bias=False)(x)
            x = layers.GroupNormalization(groups=min(out_ch // 4, 32))(x)
            x = GLU(out_ch)(x)
        else:
            x = ConvBlock(in_ch, out_ch, kernel_size, stride=stride)(x)
        skips.append(x)

    bottleneck = ConvBlock(out_ch, out_ch, kernel_size)(x)

    x = bottleneck
    for i in range(n_levels - 1, -1, -1):
        out_ch = 1 if i == 0 else encoder_channels[i - 1]
        in_ch = encoder_channels[i]
        if i == 0:
            x = layers.Conv1DTranspose(1, kernel_size, strides=1, padding="same")(x)
        else:
            x = TransposeBlock(in_ch, out_ch, kernel_size)(x, skips[i - 1])

    outputs = layers.Add()([inputs, x])

    return models.Model(inputs, outputs)
