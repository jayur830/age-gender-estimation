from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import Model, model_from_json
from keras.layers import Input, MaxPool2D, Flatten, Dense
from model.custom_layer import SkipLayer
import json
import os


def age_predictor():
    optimizer = Adam(0.05)
    loss = "categorical_crossentropy"
    metrics = ["accuracy"]

    if not os.path.exists("age_predictor_config.json") \
        and not os.path.exists("age_predictor_weights.hdf5"):
        regularizer = l2(0.001)

        # (63, 63)
        input_layer = Input(shape=(63, 63, 3))
        # (63, 63) -> (60, 60)
        net = SkipLayer(filters=8, kernel_size=(4, 4), kernel_regularizer=regularizer)(input_layer)
        # (60, 60) -> (56, 56)
        net = SkipLayer(filters=8, kernel_size=(5, 5), kernel_regularizer=regularizer)(net)
        # (56, 56) -> (28, 28)
        net = MaxPool2D()(net)

        # (28, 28) -> (26, 26)
        net = SkipLayer(filters=16, kernel_size=(3, 3), kernel_regularizer=regularizer)(net)
        # (26, 26) -> (24, 24)
        net = SkipLayer(filters=16, kernel_size=(3, 3), kernel_regularizer=regularizer)(net)
        # (24, 24) -> (12, 12)
        net = MaxPool2D()(net)

        # (12, 12) -> (10, 10)
        net = SkipLayer(filters=32, kernel_size=(3, 3), kernel_regularizer=regularizer)(net)
        # (10, 10) -> (8, 8)
        net = SkipLayer(filters=32, kernel_size=(3, 3), kernel_regularizer=regularizer)(net)
        # (8, 8) -> (4, 4)
        net = MaxPool2D()(net)

        output = Flatten()(net)
        output = Dense(
            units=6,
            kernel_initializer="he_normal",
            kernel_regularizer=regularizer,
            name="pred_age")(output)

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
                metrics=metric)
            return model
