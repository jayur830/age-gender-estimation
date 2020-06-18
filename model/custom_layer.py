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
        _net = SeparableConv2D(
            filters=self.__filters,
            kernel_size=self.__kernel_size,
            kernel_initializer=self.__kernel_initializer,
            kernel_regularizer=self.__kernel_regularizer)(inputs)
        _net = BatchNormalization()(_net)
        _skip_conn = _net = LeakyReLU()(_net)
        _net = SeparableConv2D(
            filters=self.__filters,
            kernel_size=(1, 1),
            kernel_initializer=self.__kernel_initializer,
            kernel_regularizer=self.__kernel_regularizer)(_net)
        _net = BatchNormalization()(_net)
        _net = LeakyReLU()(_net)
        _net = SeparableConv2D(
            filters=self.__filters,
            kernel_size=(3, 3),
            kernel_initializer=self.__kernel_initializer,
            kernel_regularizer=self.__kernel_regularizer)(_net)
        _net = ZeroPadding2D()(_net)
        _net = BatchNormalization()(_net)
        _net = LeakyReLU()(_net)
        _net = SeparableConv2D(
            filters=self.__filters,
            kernel_size=(1, 1),
            kernel_initializer=self.__kernel_initializer,
            kernel_regularizer=self.__kernel_regularizer)(_net)
        _net = BatchNormalization()(_net)
        _net = LeakyReLU()(_net)

        return Concatenate()([_net, _skip_conn])
