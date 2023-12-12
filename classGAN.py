from __future__ import print_function, division
import os
import numpy as np
import matplotlib.pyplot as plt

# import keras
# from tensorflow.keras.layers import Input, Dense
from keras.layers import *
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Add
from keras.applications import VGG19

from LoadData import DataLoader
import datetime

class GAN():
    def __init__(self):
        # Input shape
        self.lr_height = 64
        self.lr_width = 64
        self.channels = 3
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height * 4
        self.hr_width = self.lr_width * 4
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)
        self.n_residual_blocks = 16

        # 设置优化器
        optimizer = Adam(0.0002, 0.5)

        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        self.dataset_name = './datasets/img_align_celeba'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))

        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        self.gf = 64
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(
                                    loss='mse',
                                    optimizer=optimizer,
                                    metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)
        fake_hr = self.generator(img_lr)
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator
        # 对于组合模型，我们将只训练生成器
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        validity = self.discriminator(fake_hr)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        # 组合模型(堆叠生成器和鉴别器)
        # 训练生成器以糊弄鉴别器
        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)

    def build_vgg(self):
        print("start loading trained weights of vgg")
        vgg = VGG19(weights="imagenet")
        print("loading complete")
        vgg.outputs = [vgg.layers[9].output]

        img = Input(shape=self.hr_shape)
        img_features = vgg(img)
        return  Model(img, img_features)


    # 构造生成器
    def build_generator(self):
        def residual_block(layer_input, filters):
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        img_lr = Input(shape=self.lr_shape)

        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        r = residual_block(c1, self.gf)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, self.gf)

        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)
        return Model(img_lr, gen_hr)

    # 构造鉴别器
    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides=2)
        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides=2)
        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides=2)

        d9 = Dense(self.df * 16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        start_time = datetime.datetime.now()

        for epoch in range(epochs):
            if epoch > 30:
                sample_interval = 10
            if epoch > 100:
                sample_interval = 50

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,) + self.disc_patch)
            fake = np.zeros((batch_size,) + self.disc_patch)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            if epoch % 500 == 0 and epoch > 1:
                self.generator.save_weights('./premodel/' + str(epoch) + '.h5')

    def use_image(self, batch_size=1):
        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size, is_pred=True)
        os.makedirs('premodel/', exist_ok=True)
        self.generator.load_weights('./premodel/' + str(2000) + '.h5')
        fake_hr = self.generator.predict(imgs_lr)
        r, c = imgs_hr.shape[0], 2
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        titles = ['LR_input', 'SR_generate']
        fig, axs = plt.subplots(r, c)
        for i in range(r):
            for j, image in enumerate([imgs_lr, fake_hr]):
                axs[i, j].imshow(image[i])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')

        fig.savefig("./result.png")
        plt.close()


    def sample_images(self, epoch):
        os.makedirs('images/', exist_ok=True)
        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=1, is_testing=True, is_pred=True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5
        r, c = imgs_hr.shape[0], 3

        titles = ['LRimage', 'Generated  epoch: ' + str(epoch), 'Original']
        fig, axs = plt.subplots(r, c)

        for i in range(r):
            # for j, image in enumerate([fake_hr, imgs_hr, imgs_lr]):
            for j, image in enumerate([imgs_lr, fake_hr, imgs_hr]):
                axs[i, j].imshow(image[i])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
        fig.savefig("images/%d.png" % (epoch))
        plt.close()
