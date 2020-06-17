from keras import optimizers, regularizers
from keras.utils import plot_model
from keras.models import Model, model_from_json
from keras.layers import Input, \
    SeparableConv2D, MaxPool2D, ZeroPadding2D, \
    BatchNormalization, LeakyReLU, \
    Flatten, Dense, concatenate
import json
import os


def age_predictor():
    if not os.path.exists("age_predictor_config.json") \
        and not os.path.exists("age_predictor_weights.hdf5"):
        weight_init = "he_normal"
        LAMBDA = 0.001

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

                return concatenate([_net, _skip_conn])

        # (200, 200)
        input_layer = Input(shape=(200, 200, 3))
        # (200, 200) -> (195, 195)
        net = SkipLayer(filters=16, kernel_size=(6, 6))(input_layer)
        # (195, 195) -> (190, 190)
        net = SkipLayer(filters=16, kernel_size=(6, 6))(net)
        # (190, 190) -> (95, 95)
        net = MaxPool2D()(net)
        # (95, 95) -> (92, 92)
        net = SkipLayer(filters=32, kernel_size=(4, 4))(net)
        # (92, 92) -> (90, 90)
        net = SkipLayer(filters=32, kernel_size=(3, 3))(net)
        # (90, 90) -> (45, 45)
        net = MaxPool2D()(net)
        # (45, 45) -> (42, 42)
        net = SkipLayer(filters=64, kernel_size=(4, 4))(net)
        # (42, 42) -> (40, 40)
        net = SkipLayer(filters=64, kernel_size=(3, 3))(net)
        # (40, 40) -> (20, 20)
        net = MaxPool2D()(net)
        # (20, 20) -> (18, 18)
        net = SkipLayer(filters=128, kernel_size=(3, 3))(net)
        # (18, 18) -> (16, 16)
        net = SkipLayer(filters=128, kernel_size=(3, 3))(net)
        # (16, 16) -> (8, 8)
        net = MaxPool2D()(net)
        # (8, 8) -> (6, 6)
        net = SkipLayer(filters=256, kernel_size=(3, 3))(net)
        # (6, 6) -> (4, 4)
        net = SkipLayer(filters=256, kernel_size=(3, 3))(net)
        # (4, 4) -> (2, 2)
        net = MaxPool2D()(net)
        net = Flatten()(net)
        output = Dense(
            units=6,
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2(LAMBDA),
            name="pred_age")(net)

        model = Model(
            input_layer, output,
            name="age_predictor")
        model.summary()
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.05),
            loss="categorical_crossentropy",
            metrics=["accuracy"])
        return model
    else:
        with open("age_predictor_config.json", "r") as config:
            model = model_from_json(json.dumps(json.load(config)))
            model.load_weights("age_predictor_weights.hdf5")
            model.summary()
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.01),
                loss="categorical_crossentropy",
                metrics=["accuracy"])
            return model


if __name__ == "__main__":
    # model = age_predictor()
    # print(model.output)
    # plot_model(model, show_shapes=True)
    plot_model(age_predictor(), to_file="age_predictor.png", show_shapes=True)
