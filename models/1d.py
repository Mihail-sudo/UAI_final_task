from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Input, BatchNormalization, Activation, MaxPooling1D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def create_model():
    conv_input = Input(shape=(None, 1))

    # first block
    x = Conv1D(16, 3, padding="same")(conv_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(16, 3, padding="same")(x)
    x = BatchNormalization()(x)

    block_1_out = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(block_1_out)

    # scnd block
    x = Conv1D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    block_2_out = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(block_2_out)

    # third block
    x = Conv1D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)

    block_3_out = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(block_3_out)

    # fourth block
    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)

    block_4_out = Activation("relu")(x)
    x = MaxPooling1D(pool_size=2)(block_4_out)

    # middle block
    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # x = MaxPooling1D(pool_size=3)(x)

    # first up
    x = Conv1DTranspose(128, 2, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = concatenate([x, block_4_out])
    x = Conv1D(128, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second up
    x = Conv1DTranspose(64, 2, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = concatenate([x, block_3_out])
    x = Conv1D(64, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # third up
    x = Conv1DTranspose(32, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = concatenate([x, block_2_out])
    x = Conv1D(32, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # fourth up
    x = Conv1DTranspose(16, 3, strides=2, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = concatenate([x, block_1_out])
    x = Conv1D(16, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv1D(4, 1, padding='same')(x)

    model = Model(conv_input, x)                                             # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='mse',
                  metrics=['mse'])

    # Возвращаем сформированную модель
    return model