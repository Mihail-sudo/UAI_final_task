import tensorflow as tf
from tensorflow.keras import layers, models


def conv_block(x, filters, kernel_size=3):
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x


def create_unet(input_shape=(None, None, 1), n_filters=16, n_levels=3, n_outputs=4):
    inputs = layers.Input(shape=input_shape)

    x = inputs
    skips = []

    for i in range(n_levels):
        filters = n_filters * (2 ** i)
        x = conv_block(x, filters)
        x = conv_block(x, filters)
        skips.append(x)
        x = layers.MaxPooling2D((2, 2))(x)

    filters = n_filters * (2 ** n_levels)
    x = conv_block(x, filters)
    x = conv_block(x, filters)

    for i in range(n_levels - 1, -1, -1):
        filters = n_filters * (2 ** i)
        x = layers.Conv2DTranspose(filters, 2, strides=(2, 2), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.concatenate([x, skips[i]])
        x = conv_block(x, filters)
        x = conv_block(x, filters)

    outputs = layers.Conv2D(n_outputs, 1, padding="same", activation="sigmoid")(x)

    return models.Model(inputs, outputs)
