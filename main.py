from classGAN import GAN

gan =  GAN()
gan.train(epochs=10, batch_size=10, sample_interval=1)
gan.use_image()