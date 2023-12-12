from ClassGAN import GAN

gan =  GAN()
gan.train(epochs=81, batch_size=10, sample_interval=1)
gan.use_image()