from train import WGANGP

wgan = WGANGP()
wgan.train(epochs=30000, batch_size=32, sample_interval=10)