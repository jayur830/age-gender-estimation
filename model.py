from keras import optimizers, regularizers
from keras.utils import plot_model
from keras.models import Model, load_model, save_model
from keras.layers import Input, SeparableConv2D, MaxPool2D, ZeroPadding2D, \
    BatchNormalization, Dense, \
    Flatten, LeakyReLU, concatenate
import json


def create_model():
    weight_init = "he_normal"
    LAMBDA = 0.001

    # (63, 63)
    input_layer = Input(shape=(63, 63, 3))
    # (63, 63) -> (60, 60)
    net = SeparableConv2D(
        filters=32,
        kernel_size=(4, 4),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA),
        input_shape=(63, 63))(input_layer)
    net = BatchNormalization()(net)
    skip_conn1 = net = LeakyReLU()(net)
    # (60, 60) -> (60, 60)
    net = ZeroPadding2D()(net)
    net = SeparableConv2D(
        filters=32,
        kernel_size=(3, 3),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = concatenate([net, skip_conn1])
    # (60, 60) -> (30, 30)
    net = MaxPool2D()(net)
    # (30, 30) -> (28, 28)
    net = SeparableConv2D(
        filters=64,
        kernel_size=(3, 3),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    skip_conn2 = net = LeakyReLU()(net)
    # (28, 28) -> (28, 28)
    net = ZeroPadding2D()(net)
    net = SeparableConv2D(
        filters=64,
        kernel_size=(3, 3),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = concatenate([net, skip_conn2])
    # (28, 28) -> (14, 14)
    net = MaxPool2D()(net)
    # (14, 14) -> (10, 10)
    net = SeparableConv2D(
        filters=128,
        kernel_size=(5, 5),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    skip_conn3 = net = LeakyReLU()(net)
    # (10, 10) -> (10, 10)
    net = ZeroPadding2D()(net)
    net = SeparableConv2D(
        filters=128,
        kernel_size=(3, 3),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = concatenate([net, skip_conn3])
    # (10, 10) -> (5, 5)
    net = MaxPool2D()(net)
    # (5, 5) -> (2, 2)
    net = SeparableConv2D(
        filters=256,
        kernel_size=(4, 4),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    skip_conn4 = net = LeakyReLU()(net)
    # (2, 2) -> (2, 2)
    net = ZeroPadding2D()(net)
    net = SeparableConv2D(
        filters=256,
        kernel_size=(3, 3),
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = concatenate([net, skip_conn4])
    net = Flatten()(net)
    net = Dense(
        units=256,
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)
    net = Dense(
        units=64,
        kernel_initializer=weight_init,
        kernel_regularizer=regularizers.l2(LAMBDA))(net)
    net = BatchNormalization()(net)
    net = LeakyReLU()(net)

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
    # model.compile(
    #     optimizer=optimizers.Adam(learning_rate=0.01),
    #     loss=["categorical_crossentropy", "categorical_crossentropy"],
    #     metrics=["accuracy"])

    return model

import numpy as np
from keras.utils import to_categorical

# if __name__ == "__main__":
#     model = load_model("checkpoint/08-1.34.h5")
#     model.save_weights("model_weights.hdf5")
#     with open("model_config.json", "w") as json_file:
#         json_file.write(model.to_json())
