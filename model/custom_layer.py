from keras.regularizers import l2
from keras.layers import SeparableConv2D, ZeroPadding2D, \
    BatchNormalization, LeakyReLU, Concatenate


class SkipLayer:
    def __init__(self,
                 filters=32,
                 kernel_size=(3, 3),
                 kernel_initializer="he_normal",
                 kernel_regularizer=l2()):
        self.__filters = filters
        self.__kernel_size = kernel_size
        self.__kernel_initializer = kernel_initializer
        self.__kernel_regularizer = kernel_regularizer

    def __call__(self, inputs):
        net = SeparableConv2D(
            filters=self.__filters,
            kernel_size=self.__kernel_size,
            kernel_initializer=self.__kernel_initializer,
            kernel_regularizer=self.__kernel_regularizer)(inputs)
        net = BatchNormalization()(net)
        skip_conn = net = LeakyReLU()(net)
        net = SeparableConv2D(
            filters=self.__filters,
            kernel_size=(1, 1),
            kernel_initializer=self.__kernel_initializer,
            kernel_regularizer=self.__kernel_regularizer)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = SeparableConv2D(
            filters=self.__filters,
            kernel_size=(3, 3),
            kernel_initializer=self.__kernel_initializer,
            kernel_regularizer=self.__kernel_regularizer)(net)
        net = ZeroPadding2D()(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)
        net = SeparableConv2D(
            filters=self.__filters,
            kernel_size=(1, 1),
            kernel_initializer=self.__kernel_initializer,
            kernel_regularizer=self.__kernel_regularizer)(net)
        net = BatchNormalization()(net)
        net = LeakyReLU()(net)

        return Concatenate()([net, skip_conn])
