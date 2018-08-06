from keras import Model
from keras.layers import Input, BatchNormalization, Conv2DTranspose, Conv2D, TimeDistributed, Lambda, Activation
import keras.backend as K
from keras.utils import get_custom_objects


class Swish(Activation):
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
  return (K.sigmoid(x)*x)
get_custom_objects().update({'swish': Swish(swish)})


def falling_detection_model():
    """
    input shape: (10, 240, 320, 1)
    10 frames picture for model.

    :return: keras model
    """

    input_img = Input(shape=(10, 240, 320, 1))
    x = TimeDistributed(Conv2D(32, (11, 11), padding='same', strides=2, activation='swish'))(input_img)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(16, (5, 5), padding='same', strides=2, activation='swish'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(1, (5, 5), padding='same', strides=1, activation='swish'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Lambda(lambda i: K.squeeze(i, axis=-1)))(x)

    x = Conv2D(32, (3, 3), padding='same', data_format='channels_first', strides=2, activation='swish')(x)
    x = BatchNormalization()(x)

    x = Conv2D(16, (2, 2), padding='same', data_format='channels_first', strides=2, activation='swish')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(16, (2, 2), padding='same', data_format='channels_first', strides=2, activation='swish')(x)
    x = BatchNormalization()(x)

    x = Conv2DTranspose(32, (3, 3), padding='same', data_format='channels_first', strides=2, activation='swish')(x)
    x = BatchNormalization()(x)

    x = Conv2D(10, (3, 3), padding='same', data_format='channels_first', strides=1, activation='swish')(x)
    x = BatchNormalization()(x)

    x = TimeDistributed(Lambda(lambda i: K.expand_dims(i, axis=-1)))(x)
    x = TimeDistributed(Conv2DTranspose(16, (5, 5), padding='same', strides=2, activation='swish'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2DTranspose(32, (11, 11), padding='same', strides=2, activation='swish'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Conv2D(1, (11, 11), activation='sigmoid', padding='same', strides=1))(x)

    model = Model(input_img, x)
    model.compile(optimizer='adam', loss='mse')

    return model