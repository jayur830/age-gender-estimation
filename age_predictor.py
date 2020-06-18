from keras import optimizers, regularizers
from keras.utils import plot_model
from keras.models import Model, model_from_json
from keras.layers import Input, \
    SeparableConv2D, MaxPool2D, ZeroPadding2D, \
    BatchNormalization, LeakyReLU, \
    Flatten, Dense, Concatenate
import json
import os


def age_predictor():
    if not os.path.exists("age_predictor_config.json") \
        and not os.path.exists("age_predictor_weights.hdf5"):
        weight_init = "he_normal"
        LAMBDA = 0.01

        class SkipLayer:
            def __init__(self, filters=32, kernel_size=(3, 3)):
                self.__filters = filters
                self.__kernel_size = kernel_size

            def __call__(self, inputs):
                _net = SeparableConv2D(
                    filters=self.__filters,
                    kernel_size=self.__kernel_size,
                    kernel_initializer=weight_init,
                    kernel_regularizer=regularizers.l2(LAMBDA))(inputs)
                _net = BatchNormalization()(_net)
                _skip_conn = _net = LeakyReLU()(_net)
                _net = SeparableConv2D(
                    filters=self.__filters,
                    kernel_size=(1, 1),
                    kernel_initializer=weight_init,
                    kernel_regularizer=regularizers.l2(LAMBDA))(_net)
                _net = BatchNormalization()(_net)
                _net = LeakyReLU()(_net)
                _net = SeparableConv2D(
                    filters=self.__filters,
                    kernel_size=(3, 3),
                    kernel_initializer=weight_init,
                    kernel_regularizer=regularizers.l2(LAMBDA))(_net)
                _net = ZeroPadding2D()(_net)
                _net = BatchNormalization()(_net)
                _net = LeakyReLU()(_net)
                _net = SeparableConv2D(
                    filters=self.__filters,
                    kernel_size=(1, 1),
                    kernel_initializer=weight_init,
                    kernel_regularizer=regularizers.l2(LAMBDA))(_net)
                _net = BatchNormalization()(_net)
                _net = LeakyReLU()(_net)

                return Concatenate()([_net, _skip_conn])

        # (63, 63)
        input_layer = Input(shape=(63, 63, 3))
        # (63, 63) -> (60, 60)
        net = SkipLayer(filters=32, kernel_size=(4, 4))(input_layer)
        # (60, 60) -> (56, 56)
        net = SkipLayer(filters=32, kernel_size=(5, 5))(net)
        # (56, 56) -> (28, 28)
        net = MaxPool2D()(net)

        # (28, 28) -> (26, 26)
        net = SkipLayer(filters=64, kernel_size=(3, 3))(net)
        # (26, 26) -> (24, 24)
        net = SkipLayer(filters=64, kernel_size=(3, 3))(net)
        # (24, 24) -> (12, 12)
        net = MaxPool2D()(net)

        # (12, 12) -> (10, 10)
        net = SkipLayer(filters=128, kernel_size=(3, 3))(net)
        # (10, 10) -> (8, 8)
        net = SkipLayer(filters=128, kernel_size=(3, 3))(net)
        # (8, 8) -> (4, 4)
        net = MaxPool2D()(net)

        output = Flatten()(net)
        output = Dense(
            units=6,
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2(LAMBDA),
            name="pred_age")(output)

        model = Model(
            input_layer, output,
            name="age_predictor")
        model.summary()
        model.compile(
            optimizer=optimizers.Adam(lr=0.01),
            loss="categorical_crossentropy",
            metrics=["accuracy"])
        return model
    else:
        with open("age_predictor_config.json", "r") as config:
            model = model_from_json(json.dumps(json.load(config)))
            model.load_weights("age_predictor_weights.hdf5")
            model.summary()
            model.compile(
                optimizer=optimizers.Adam(lr=0.01),
                loss="categorical_crossentropy",
                metrics=["accuracy"])
            return model


if __name__ == "__main__":
    # model = age_predictor()
    # print(model.output)
    # plot_model(model, show_shapes=True)
    plot_model(age_predictor(), to_file="_age_predictor.png", show_shapes=True)
