#implement Generator

#######################################################################

from __future__ import print_function, division
from keras.layers import Input, Dense, Reshape, Flatten, Dropout,Embedding,multiply,LeakyReLU
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import numpy as np

#######################################################################

def build_generator():

    ##########################################
    channels = 1
    num_classes=10
    latent_dim = 100
    num_classes = 10
    img_rows = 28
    img_cols = 28
    img_shape = (img_rows, img_cols, channels)
    ##########################################

    model = Sequential()

    # model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    # model.add(Reshape((7, 7, 128)))
    # model.add(UpSampling2D())
    # model.add(Conv2D(128, kernel_size=4, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))
    # model.add(UpSampling2D())
    # model.add(Conv2D(64, kernel_size=4, padding="same"))
    # model.add(BatchNormalization(momentum=0.8))
    # model.add(Activation("relu"))
    # model.add(Conv2D(channels, kernel_size=4, padding="same"))
    # model.add(Activation("tanh"))
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    model.summary()

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))

    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)