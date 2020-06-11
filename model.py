from keras import optimizers, regularizers
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, \
    BatchNormalization, Dense, \
    Flatten, LeakyReLU


def create_model():
    dsize = (63, 63)
    weight_init = "he_normal"
    LAMBDA = 0.01

    # (63, 63)
    input_layer = Input(shape=(dsize[0], dsize[1], 3))
    # (63, 63) -> (60, 60)
    net = Conv2D(
        filters=10,
        kernel_size=(4, 4),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA),
        input_shape=dsize)(input_layer)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    # (60, 60) -> (58, 58)
    net = Conv2D(
        filters=30,
        kernel_size=(3, 3),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    # (58, 58) -> (26, 26)
    net = MaxPool2D()(net)
    # (26, 26) -> (23, 23)
    net = Conv2D(
        filters=50,
        kernel_size=(4, 4),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    # (23, 23) -> (20, 20)
    net = Conv2D(
        filters=70,
        kernel_size=(4, 4),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    # (20, 20) -> (10, 10)
    net = MaxPool2D()(net)
    # (10, 10) -> (8, 8)
    net = Conv2D(
        filters=100,
        kernel_size=(3, 3),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    # (8, 8) -> (6, 6)
    net = Conv2D(
        filters=150,
        kernel_size=(3, 3),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    # (6, 6) -> (4, 4)
    net = Conv2D(
        filters=200,
        kernel_size=(3, 3),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    # (4, 4) -> (2, 2)
    net = MaxPool2D()(net)
    net = Flatten()(net)
    net = Dense(
        units=256,
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = Dense(
        units=128,
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

    gender_predictor = Dense(units=2, activation="softmax", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.01), name="pred_gender")(net)
    age_predictor = Dense(units=60, activation="softmax", kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.01), name="pred_age")(net)

    model = Model(input_layer, [gender_predictor, age_predictor], name="age_gender_estimation")
    model.summary()
    model.compile(optimizer=optimizers.Adam(learning_rate=0.01), loss=["binary_crossentropy", "categorical_crossentropy"], metrics=["accuracy"])

    return model
