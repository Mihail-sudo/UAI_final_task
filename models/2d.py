from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, Input, BatchNormalization, Activation, MaxPooling2D, concatenate
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def conv_block(x, filters):
    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def create_model():
    inputs = Input(shape=(None, None, 1))

    # Encoder
    c1 = conv_block(inputs, 16)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 32)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 64)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 128)
    p4 = MaxPooling2D((2, 2))(c4)

    # Bottleneck
    b = conv_block(p4, 256)

    # Decoder
    u1 = Conv2DTranspose(
        128,
        kernel_size=2,
        strides=2,
        padding="same"
    )(b)

    u1 = concatenate([u1, c4])

    c5 = conv_block(u1, 128)

    u2 = Conv2DTranspose(
        64,
        kernel_size=2,
        strides=2,
        padding="same"
    )(c5)

    u2 = concatenate([u2, c3])

    c6 = conv_block(u2, 64)

    u3 = Conv2DTranspose(
        32,
        kernel_size=2,
        strides=2,
        padding="same"
    )(c6)

    u3 = concatenate([u3, c2])

    c7 = conv_block(u3, 32)

    u4 = Conv2DTranspose(
        16,
        kernel_size=2,
        strides=2,
        padding="same"
    )(c7)

    u4 = concatenate([u4, c1])

    c8 = conv_block(u4, 16)

    outputs = Conv2D(
        4,
        kernel_size=1,
        activation="sigmoid",
        padding="same"
    )(c8)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(1e-4),
        loss="mse",
        metrics=["mse"]
    )
    return model