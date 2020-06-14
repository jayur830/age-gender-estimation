from keras import optimizers, regularizers
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Input, SeparableConv2D, MaxPool2D, \
    BatchNormalization, Dense, \
    Flatten, ReLU, concatenate


def create_model():
    weight_init = "he_normal"
    LAMBDA = 0.001

    # (63, 63)
    input_layer = Input(shape=(63, 63, 3))
    # (63, 63) -> (60, 60)
    net = SeparableConv2D(
        filters=10,
        kernel_size=(4, 4),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA),
        input_shape=(63, 63))(input_layer)
    net = BatchNormalization()(net)
    skip_conn = net = ReLU()(net)
    # (60, 60) -> (60, 60)
    net = SeparableConv2D(
        filters=30,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    net = concatenate([net, skip_conn])
    # (60, 60) -> (30, 30)
    net = MaxPool2D(strides=2)(net)
    # (30, 30) -> (28, 28)
    net = SeparableConv2D(
        filters=50,
        kernel_size=(3, 3),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    skip_conn = net = ReLU()(net)
    # (28, 28) -> (28, 28)
    net = SeparableConv2D(
        filters=70,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    net = concatenate([net, skip_conn])
    # (28, 28) -> (14, 14)
    net = MaxPool2D(strides=2)(net)
    # (14, 14) -> (10, 10)
    net = SeparableConv2D(
        filters=100,
        kernel_size=(5, 5),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    skip_conn = net = ReLU()(net)
    # (10, 10) -> (10, 10)
    net = SeparableConv2D(
        filters=150,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    net = concatenate([net, skip_conn])
    # (10, 10) -> (5, 5)
    net = MaxPool2D(strides=2)(net)
    # (5, 5) -> (2, 2)
    net = SeparableConv2D(
        filters=200,
        kernel_size=(4, 4),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    skip_conn = net = ReLU()(net)
    # (2, 2) -> (2, 2)
    net = SeparableConv2D(
        filters=250,
        kernel_size=(3, 3),
        padding="same",
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    net = concatenate([net, skip_conn])
    net = Flatten()(net)
    net = Dense(
        units=128,
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)
    net = Dense(
        units=64,
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = ReLU()(net)

    gender_predictor = Dense(
        units=2,
        activation="softmax",
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA),
        name="pred_gender")(net)
    age_predictor = Dense(
        units=6,
        activation="softmax",
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA),
        name="pred_age")(net)

    model = Model(
        input_layer,
        [gender_predictor, age_predictor],
        name="age_gender_estimation")
    model.summary()
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.01),
        loss=["binary_crossentropy", "categorical_crossentropy"],
        metrics=["accuracy"])

    return model
