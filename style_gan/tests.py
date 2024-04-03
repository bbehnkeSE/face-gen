import torch
from   torch import nn

from   model import *

import math

if __name__ == "__main__":
    # print("Testing MappingLayers...")
    # map = MappingLayers(10, 20, 30)
    # assert tuple(map(torch.randn(2, 10)).shape) == (2, 30)
    # assert len(map.map) > 4
    # w = map(torch.randn(1000, 10))
    # assert w.std() > 0.05 and w.std() < 0.3
    # assert w.min() > -2   and w.min() < 0
    # assert w.max() < 2    and w.max() > 0
    # print("Done.")


    # print("Testing InjectNoise...")
    # test_noise_channels = 3000
    # test_noise_samples = 20
    # fake_images = torch.randn(test_noise_samples, test_noise_channels, 10, 10)
    # inject_noise = InjectNoise(test_noise_channels)
    # assert torch.abs(inject_noise.weight.std() - 1) < 0.1
    # assert torch.abs(inject_noise.weight.mean()) < 0.1
    # assert type(inject_noise.weight) == torch.nn.parameter.Parameter
    # assert tuple(inject_noise.weight.shape) == (1, test_noise_channels, 1, 1)
    # inject_noise.weight = nn.Parameter(torch.ones_like(inject_noise.weight))

    # # Check that something changed
    # assert torch.abs((inject_noise(fake_images) - fake_images)).mean() > 0.1

    # # Check that the change is per-channel
    # assert torch.abs((inject_noise(fake_images) - fake_images).std(0)).mean() > 1e-4
    # assert torch.abs((inject_noise(fake_images) - fake_images).std(1)).mean() < 1e-4
    # assert torch.abs((inject_noise(fake_images) - fake_images).std(2)).mean() > 1e-4
    # assert torch.abs((inject_noise(fake_images) - fake_images).std(3)).mean() > 1e-4

    # # Check that the per-channel change is roughly normal
    # per_channel_change = (inject_noise(fake_images) - fake_images).mean(1).std()
    # assert per_channel_change > 0.9 and per_channel_change < 1.1

    # # Make sure that the weights are being used at all
    # inject_noise.weight = nn.Parameter(torch.zeros_like(inject_noise.weight))
    # assert torch.abs((inject_noise(fake_images) - fake_images)).mean() < 1e-4
    # assert len(inject_noise.weight.shape) == 4
    # print("Done.")


    # print("Testing AdaIN...")
    # w_channels = 50
    # image_channels = 20
    # image_size = 30
    # n_test = 10
    # adain = AdaIN(image_channels, w_channels)
    # test_w = torch.randn(n_test, w_channels)
    # assert adain.style_scale(test_w).shape == adain.style_bias(test_w).shape
    # assert adain.style_scale(test_w).shape[-1] == image_channels
    # assert tuple(adain(torch.randn(n_test, image_channels, image_size, image_size), test_w).shape) == (n_test, image_channels, image_size, image_size)

    # w_channels = 3
    # image_channels = 2
    # image_size = 3
    # n_test = 1
    # adain = AdaIN(image_channels, w_channels)

    # adain.style_scale.weight.data = torch.ones_like(adain.style_scale.weight.data) / 4
    # adain.style_scale.bias.data = torch.zeros_like(adain.style_scale.bias.data)
    # adain.style_bias.weight.data = torch.ones_like(adain.style_bias.weight.data) / 5
    # adain.style_bias.bias.data = torch.zeros_like(adain.style_bias.bias.data)
    # test_input = torch.ones(n_test, image_channels, image_size, image_size)
    # test_input[:, :, 0] = 0
    # test_w = torch.ones(n_test, w_channels)
    # test_output = adain(test_input, test_w)
    # assert(torch.abs(test_output[0, 0, 0, 0] - 3 / 5 + torch.sqrt(torch.tensor(9 / 8))) < 1e-4)
    # assert(torch.abs(test_output[0, 0, 1, 0] - 3 / 5 - torch.sqrt(torch.tensor(9 / 32))) < 1e-4)
    # print("Done.")


    # print("Testing GenBlock...")
    # test_stylegan_block = GenBlock(in_chan=128, out_chan=64, w_dim=256, kernel_size=3, starting_size=8)
    # test_x = torch.ones(1, 128, 4, 4)
    # test_x[:, :, 1:3, 1:3] = 0
    # test_w = torch.ones(1, 256)
    # test_x = test_stylegan_block.upsample(test_x)
    # assert tuple(test_x.shape) == (1, 128, 8, 8)
    # assert torch.abs(test_x.mean() - 0.75) < 1e-4
    # test_x = test_stylegan_block.conv(test_x)
    # assert tuple(test_x.shape) == (1, 64, 8, 8)
    # test_x = test_stylegan_block.inject_noise(test_x)
    # test_x = test_stylegan_block.activation(test_x)
    # assert test_x.min() < 0
    # assert -test_x.min() / test_x.max() < 0.4
    # test_x = test_stylegan_block.adain(test_x, test_w) 
    # foo = test_stylegan_block(torch.ones(10, 128, 4, 4), torch.ones(10, 256))
    # print("Done.")


    print("Testing GAN...")
    Z_DIM = 512
    W_DIM = 512
    IN_CHANNELS = 512
    gen = Generator(Z_DIM, W_DIM, IN_CHANNELS).to("cuda")
    disc = Critic(IN_CHANNELS, img_chan=3).to("cuda")

    tot = 0
    for param in gen.parameters():
        tot += param.numel()

    print(tot)

    for img_size in [8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(math.log2(img_size / 4))
        x = torch.randn((2, Z_DIM)).to("cuda")
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (2, 3, img_size, img_size), f"{z.shape = }"
        out = disc(z, alpha=0.5, steps=num_steps)
        assert out.shape == (2, 1)
        print(f"Success! At img size: {img_size}")
    print("Done.")