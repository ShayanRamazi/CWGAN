#implement Critic

#######################################################################

from __future__ import print_function, division
from keras.layers import Input, Dense, Flatten, Dropout,Embedding,multiply
from keras.layers import BatchNormalization,  ZeroPadding2D,Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
import numpy as np
#######################################################################

def build_critic():

    ##########################################
    img_rows = 28
    img_cols = 28
    channels = 1
    num_classes = 10
    img_shape = (img_rows,img_cols, channels)
    ##########################################
    model = Sequential()

    # model.add(Dense(7 * 7 * 128, input_dim=np.prod(img_shape)))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Reshape((7, 7, 128)))
    # model.add(Conv2D(16, kernel_size=3, strides=2, padding="same"))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    # model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(LeakyReLU(alpha=0.2))
    # model.add(Dropout(0.25))
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1))

    model.summary()

    img = Input(shape=img_shape)
    label = Input(shape=(1,), dtype='int32')

    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    flat_img = Flatten()(img)

    model_input = multiply([flat_img, label_embedding])

    validity = model(model_input)

    return Model([img, label], validity)