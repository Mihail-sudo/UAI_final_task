import tensorflow as tf
from tensorflow.keras import layers, models
import math


def conv_block(x, filters, kernel_size=3):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.GroupNormalization(groups=min(filters // 8, 32))(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.GroupNormalization(groups=min(filters // 8, 32))(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def se_block(x, ratio=8):
    channels = x.shape[-1]
    squeeze = layers.GlobalAveragePooling2D()(x)
    squeeze = layers.Dense(max(channels // ratio, 4), activation="relu")(squeeze)
    squeeze = layers.Dense(channels, activation="sigmoid")(squeeze)
    return layers.multiply([x, squeeze])


class ECABlock(layers.Layer):
    def __init__(self, gamma=2, b=1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.b = b

    def build(self, input_shape):
        channels = input_shape[-1]
        k = max(int(abs(math.log2(channels) / self.gamma + self.b / self.gamma)), 3)
        k = k if k % 2 == 1 else k + 1
        self.gap = layers.GlobalAveragePooling2D()
        self.conv = layers.Conv1D(1, k, padding="same", use_bias=False)

    def call(self, x):
        gap = self.gap(x)[..., tf.newaxis]
        gap = self.conv(gap)
        gap = tf.sigmoid(gap)
        gap = tf.transpose(gap, [0, 2, 1])
        gap = gap[:, tf.newaxis, :, :]
        return layers.multiply([x, gap])


def eca_block(x, gamma=2, b=1):
    return ECABlock(gamma, b)(x)


class MHABottleneck(layers.Layer):
    def __init__(self, n_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads

    def build(self, input_shape):
        c = input_shape[-1]
        self.mha = layers.MultiHeadAttention(
            num_heads=self.n_heads, key_dim=max(c // self.n_heads // 2, 16)
        )
        self.ln = layers.LayerNormalization()

    def call(self, x):
        b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[-1]
        x_t = tf.reshape(x, [b, h * w, c])
        attn_out = self.mha(x_t, x_t)
        attn_out = tf.reshape(attn_out, [b, h, w, c])
        return self.ln(x + attn_out)


def create_unet(
    input_shape=(None, None, 1),
    n_filters=32,
    n_levels=5,
    n_outputs=8,
    dropout_rate=0.2,
    use_eca=True,
    bottleneck_attention=True,
    n_input_channels=None,
):
    if n_input_channels is not None:
        input_shape = (input_shape[0], input_shape[1], n_input_channels)
    inputs = layers.Input(shape=input_shape)

    x = inputs
    skips = []
    filter_list = [n_filters * (2 ** i) for i in range(n_levels)]

    for filters in filter_list:
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        if use_eca:
            x = eca_block(x)
        skips.append(x)
        x = layers.AveragePooling2D((2, 2))(x)

    bottleneck_filters = filter_list[-1] * 2
    x = conv_block(x, bottleneck_filters)
    x = conv_block(x, bottleneck_filters)
    if use_eca:
        x = eca_block(x)
    if bottleneck_attention:
        x = MHABottleneck(n_heads=4)(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    for i in range(n_levels - 1, -1, -1):
        filters = filter_list[i]
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.GroupNormalization(groups=min(filters // 8, 32))(x)
        x = layers.Activation("relu")(x)
        x = layers.concatenate([x, skips[i]])
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        if use_eca and i > 0:
            x = eca_block(x)
        if dropout_rate > 0 and i > 0:
            x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Conv2D(n_outputs, 1, padding="same", dtype="float32")(x)

    return models.Model(inputs, outputs)
