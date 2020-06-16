from keras import optimizers, regularizers
from keras.utils import plot_model
from keras.models import Model, model_from_json
from keras.layers import Input, \
    SeparableConv2D, MaxPool2D, ZeroPadding2D, \
    BatchNormalization, LeakyReLU, \
    Flatten, Dense, concatenate
import json
import os


def gender_predictor():
    if not os.path.exists("gender_predictor_config.json") \
        and not os.path.exists("gender_predictor_weights.hdf5"):
        weight_init = "he_normal"

        # (63, 63)
        input_layer = Input(shape=(63, 63, 3))
        # (63, 63) -> (60, 60)
        net = SeparableConv2D(
            filters=32,
            kernel_size=(4, 4),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2(),
            input_shape=(63, 63))(input_layer)
        net = BatchNormalization()(net)
        skip_conn = net = LeakyReLU()(net)
        # (60, 60) -> (60, 60)
        net = SeparableConv2D(
            filters=32,
            kernel_size=(1, 1),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2())(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        # (60, 60) -> (60, 60)
        net = SeparableConv2D(
            filters=32,
            kernel_size=(3, 3),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2())(net)
        net = ZeroPadding2D()(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        # (60, 60) -> (60, 60)
        net = SeparableConv2D(
            filters=32,
            kernel_size=(1, 1),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2())(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = concatenate([net, skip_conn])
        # (60, 60) -> (30, 30)
        net = MaxPool2D()(net)
        # (30, 30) -> (28, 28)
        net = SeparableConv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2())(net)
        net = BatchNormalization()(net)
        skip_conn = net = LeakyReLU()(net)
        # (28, 28) -> (28, 28)
        net = SeparableConv2D(
            filters=64,
            kernel_size=(1, 1),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2())(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        # (28, 28) -> (28, 28)
        net = SeparableConv2D(
            filters=64,
            kernel_size=(3, 3),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2())(net)
        net = ZeroPadding2D()(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        # (28, 28) -> (28, 28)
        net = SeparableConv2D(
            filters=64,
            kernel_size=(1, 1),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2())(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = concatenate([net, skip_conn])
        # (28, 28) -> (14, 14)
        net = MaxPool2D()(net)
        # (14, 14) -> (10, 10)
        net = SeparableConv2D(
            filters=128,
            kernel_size=(5, 5),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2())(net)
        net = BatchNormalization()(net)
        skip_conn = net = LeakyReLU()(net)
        # (10, 10) -> (10, 10)
        net = SeparableConv2D(
            filters=128,
            kernel_size=(1, 1),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2())(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        # (10, 10) -> (10, 10)
        net = SeparableConv2D(
            filters=128,
            kernel_size=(3, 3),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2())(net)
        net = ZeroPadding2D()(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        # (10, 10) -> (10, 10)
        net = SeparableConv2D(
            filters=128,
            kernel_size=(1, 1),
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2())(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = concatenate([net, skip_conn])
        # (10, 10) -> (5, 5)
        net = MaxPool2D()(net)
        net = Flatten()(net)

        gender_predictor = Dense(
            units=2,
            activation="softmax",
            kernel_initializer=weight_init,
            kernel_regularizer=regularizers.l2(),
            name="pred_gender")(net)

        model = Model(
            input_layer, gender_predictor,
            name="age_gender_estimation")

        model.load_weights("_gender_predictor_weights.hdf5")
        for layer in model.layers:
            if layer.name == "separable_conv2d_9":
                break
            else:
                layer.trainable = False

        model.summary()
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.01),
            loss="categorical_crossentropy",
            metrics=["accuracy"])
        return model
    else:
        with open("gender_predictor_config.json", "r") as config:
            model = model_from_json(json.dumps(json.load(config)))
            model.load_weights("gender_predictor_weights.hdf5")
            model.summary()
            model.compile(
                optimizer=optimizers.Adam(learning_rate=0.01),
                loss="categorical_crossentropy",
                metrics=["accuracy"])
            return model


if __name__ == "__main__":
    model = gender_predictor()
    print(model.output)
    plot_model(model, show_shapes=True)