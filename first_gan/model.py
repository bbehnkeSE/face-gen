import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim, im_dim=784, h_dim=128):
        super().__init__()
        self.generator = nn.Sequential(
            self.gen_block(z_dim, h_dim),
            self.gen_block(h_dim, h_dim*2),
            self.gen_block(h_dim*2, h_dim*4),
            self.gen_block(h_dim*4, h_dim*8),
            nn.Linear(h_dim*8, im_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        return self.generator(noise)

    def gen_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )


class Discriminator(nn.Module):
    def __init__(self, im_dim=784, h_dim=128):
        super().__init__()
        self.discriminator = nn.Sequential(
            self.disc_block(im_dim, h_dim*4),
            self.disc_block(h_dim*4, h_dim*2),
            self.disc_block(h_dim*2, h_dim),
            nn.Linear(h_dim, 1)
        )

    def forward(self, img):
        return self.discriminator(img)

    def disc_block(self, input_dim, output_dim):
        return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LeakyReLU(0.2)
        )