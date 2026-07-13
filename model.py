import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(x, filters, kernel_size=3):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
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


class SelfAttentionBlock(layers.Layer):
    def __init__(self, n_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads

    def build(self, input_shape):
        c = input_shape[-1]
        key_dim = max(c // self.n_heads, 8)
        self.mha = layers.MultiHeadAttention(
            num_heads=self.n_heads, key_dim=key_dim
        )
        self.proj = layers.Dense(c)
        self.ln = layers.LayerNormalization()

    def call(self, x):
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = x.shape[-1]

        x_t = tf.reshape(x, [b * w, h, c])
        attn_out = self.mha(x_t, x_t)
        attn_out = self.proj(attn_out)
        attn_out = tf.reshape(attn_out, [b, w, h, c])
        attn_out = tf.transpose(attn_out, [0, 2, 1, 3])

        return self.ln(x + attn_out)


def create_unet(
    input_shape=(None, None, 1),
    n_filters=16,
    n_levels=3,
    n_outputs=4,
    dropout_rate=0.3,
    use_se=True,
    se_ratio=8,
    use_self_attn=True,
    self_attn_heads=4,
    n_input_channels=None,
):
    if n_input_channels is not None:
        input_shape = (input_shape[0], input_shape[1], n_input_channels)
    inputs = layers.Input(shape=input_shape)

    x = inputs
    skips = []

    for i in range(n_levels):
        filters = n_filters * (2 ** i)
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        if use_se:
            x = se_block(x, se_ratio)
        skips.append(x)
        x = layers.MaxPooling2D((2, 2))(x)

    filters = n_filters * (2 ** n_levels)
    x = conv_block(x, filters)
    x = conv_block(x, filters)
    if use_se:
        x = se_block(x, se_ratio)
    if use_self_attn:
        x = SelfAttentionBlock(n_heads=self_attn_heads)(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    for i in range(n_levels - 1, -1, -1):
        filters = n_filters * (2 ** i)
        x = layers.Conv2DTranspose(filters, 2, strides=(2, 2), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.concatenate([x, skips[i]])
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        if use_se:
            x = se_block(x, se_ratio)
        if dropout_rate > 0 and i > 0:
            x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Conv2D(n_outputs, 1, padding="same", activation="softmax")(x)

    return models.Model(inputs, outputs)
