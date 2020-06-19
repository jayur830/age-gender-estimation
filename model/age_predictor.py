import json
import os

from keras.layers import \
    Input, Conv2D, MaxPool2D, \
    Flatten, Dense, \
    BatchNormalization, LeakyReLU
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.regularizers import l2

from model.custom_layer import SkipLayer


def age_predictor():
    optimizer = Adam(0.01)
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]

    if not os.path.exists("age_predictor_config.json") \
            and not os.path.exists("age_predictor_weights.hdf5"):
        weight_init = "he_normal"
        LAMBDA = 0.01

        '''
        Feature Extraction
        '''
        # (100, 100)
        input_layer = Input(shape=(100, 100, 1))
        # (100, 100) -> (98, 98)
        net = SkipLayer(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=l2(LAMBDA))(input_layer)
        # (98, 98) -> (96, 96)
        net = SkipLayer(
            filters=16,
            kernel_size=(3, 3),
            kernel_initializer=l2(LAMBDA))(net)
        # (96, 96) -> (48, 48)
        net = MaxPool2D()(net)
        # (48, 48) -> (46, 46)
        net = SkipLayer(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=l2(LAMBDA))(net)
        # (46, 46) -> (44, 44)
        net = SkipLayer(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=l2(LAMBDA))(net)
        # (44, 44) -> (22, 22)
        net = MaxPool2D()(net)
        # (22, 22) -> (20, 20)
        net = SkipLayer(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=l2(LAMBDA))(net)
        # (20, 20) -> (18, 18)
        net = SkipLayer(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=l2(LAMBDA))(net)
        # (18, 18) -> (9, 9)
        net = MaxPool2D()(net)
        # (9, 9) -> (7, 7)
        net = SkipLayer(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=l2(LAMBDA))(net)
        # (7, 7) -> (4, 4)
        net = SkipLayer(
            filters=128,
            kernel_size=(4, 4),
            kernel_initializer=l2(LAMBDA))(net)
        # (4, 4) -> (2, 2)
        net = MaxPool2D()(net)
        # Flatten
        net = Flatten()(net)
        '''
        Classification
        '''
        net = Dense(
            units=256,
            kernel_initializer="he_normal",
            kernel_regularizer=l2(LAMBDA))(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        output = Dense(
            units=6,
            activation="softmax",
            kernel_initializer="he_normal",
            kernel_regularizer=l2(LAMBDA),
            name="pred_age")(net)

        model = Model(
            input_layer, output,
            name="age_predictor")
        model.summary()
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics)
        return model
    else:
        with open("age_predictor_config.json", "r") as config:
            model = model_from_json(json.dumps(json.load(config)))
            model.load_weights("age_predictor_weights.hdf5")
            model.summary()
            model.compile(
                optimizer=optimizer,
                loss=loss,
                metrics=metrics)
            return model
