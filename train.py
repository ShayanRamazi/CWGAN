#implement WGANGP

#######################################################################

from __future__ import print_function, division
from keras.datasets import mnist
from LossFunctions import wasserstein_loss,gradient_penalty_loss
from keras.layers import Input
from keras.models import  Model
from keras.optimizers import RMSprop
from functools import partial
import numpy as np
from RandomWeightedAverage import RandomWeightedAverage
from Critic import build_critic
from Generator import build_generator
import matplotlib.pyplot as plt
import math
from keras.utils import plot_model

#######################################################################
class WGANGP():
    def __init__(self):

        ##########################################
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.n_critic = 5
        self.n_critic=10
        optimizer = RMSprop(lr=0.00005)
        self.generator = build_generator()
        self.critic = build_critic()
        self.batch_size=32
        self.losslog = []
        self.nclasses=10
        ##########################################

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))

        # Generate image based of noise (fake sample) and add label to the input
        label = Input(shape=(1,))
        fake_img = self.generator([z_disc, label])

        # Discriminator determines validity of the real and fake images
        fake = self.critic([fake_img, label])
        valid = self.critic([real_img, label])

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])

        # Determine validity of weighted sample
        validity_interpolated = self.critic([interpolated_img, label])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'  # Keras requires function names

        self.critic_model = Model(inputs=[real_img, label, z_disc], outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[wasserstein_loss    ,
                                        wasserstein_loss    ,
                                        partial_gp_loss]    ,
                                        optimizer=optimizer ,
                                        loss_weights=[1, 1, 10])

        # -------------------------------
        # Construct Computational Graph
        #         for Generator
        # -------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(100,))
        # add label to the input
        label = Input(shape=(1,))
        # Generate images based of noise
        img = self.generator([z_gen, label])
        # Discriminator determines validity
        valid = self.critic([img, label])
        # Defines generator model
        self.generator_model = Model([z_gen, label], valid)
        plot_model(self.generator_model,to_file="./model shape/model.pdf",show_shapes=True)
        self.generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.array(list(range(10)) * 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = self.combine_images(gen_imgs)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        plt.imshow(gen_imgs, cmap='gray')
        plt.axis('off')
        plt.savefig("images/mnist_%d.png" % epoch)
        plt.close()

    def combine_images(self, generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num) / width))
        shape = generated_images.shape[1:3]
        image = np.zeros((height * shape[0], width * shape[1]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index / width)
            j = index % width
            image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1]] = \
                img[:, :, 0]
        return image

    def generate_images(self, label):
        self.generator.load_weights('../cwgan_gp/generator')
        noise = np.random.normal(0, 1, (1, self.latent_dim))
        gen_imgs = self.generator.predict([noise, np.array(label).reshape(-1, 1)])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1

        plt.imshow(gen_imgs[0, :, :, 0], cmap='gray')
        plt.axis('off')

        plt.close()


    def train(self, epochs, batch_size, sample_interval=50):

        # Load the dataset
        (X_train, y_train), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):
            for _ in range(self.n_critic):
                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs, labels = X_train[idx], y_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, labels, noise], [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------
            sampled_labels = np.random.randint(0, self.nclasses, self.batch_size).reshape(-1, 1)
            g_loss = self.generator_model.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            self.losslog.append([d_loss[0], g_loss])

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.generator.save_weights('generator', overwrite=True)
                self.critic.save_weights('discriminator', overwrite=True)
                with open('loss.log', 'w') as f:
                    f.writelines('d_loss, g_loss\n')
                    for each in self.losslog:
                        f.writelines('%s, %s\n' % (each[0], each[1]))