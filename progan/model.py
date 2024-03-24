import torch
from   torch import nn
import math

factors = [1., 1., 1., 1., 0.5, 0.25, 0.125, 0.0625, 0.03125]

class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv2d      = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale       = (gain / (in_channels * (kernel_size ** 2))) ** 0.5
        self.bias        = self.conv2d.bias
        self.conv2d.bias = None

        nn.init.normal_(self.conv2d.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, input):
        return self.conv2d(input * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)
    

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, input):
        return input/torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + self.epsilon)
    

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_pn=True):
        super().__init__()
        self.conv1      = WSConv2d(in_channels, out_channels)
        self.conv2      = WSConv2d(out_channels, out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.pn         = PixelNorm()
        self.use_pn     = use_pn

    def forward(self, input):
        input = self.activation(self.conv1(input))
        input = self.pn(input) if self.use_pn else input
        input = self.activation(self.conv2(input))
        input = self.pn(input) if self.use_pn else input

        return input


class Generator(nn.Module):
    def __init__(self, z_dim, in_channels, img_channels=3):
        super().__init__()
        self.init_layer = nn.Sequential(
            PixelNorm(),
            nn.ConvTranspose2d(z_dim, in_channels, 4, 1, 0),  # 512 x 1 x 1 -> 512 x 4 x 4
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.init_rgb           = WSConv2d(in_channels, img_channels,kernel_size=1, padding=0)
        self.progressive_blocks = nn.ModuleList()
        self.rgb_layers         = nn.ModuleList([self.init_rgb])

        for i in range(len(factors) - 1):
            conv_in_chan  = int(in_channels * factors[i])
            conv_out_chan = int(in_channels * factors[i+1])
            self.progressive_blocks.append(Conv2dBlock(conv_in_chan, conv_out_chan))
            self.rgb_layers.append(WSConv2d(conv_out_chan, img_channels, kernel_size=1, padding=0))

    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh(alpha * generated + (1-alpha) * upscaled)

    def forward(self, input, alpha, steps):
        out = self.init_layer(input)

        if steps == 0:
            return self.init_rgb(out)
        
        for step in range(steps):
            upscaled = nn.functional.interpolate(out, scale_factor=2, mode="nearest")
            out = self.progressive_blocks[step](upscaled)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out      = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)
    

class Critic(nn.Module):
    def __init__(self, in_channels, img_channels=3):
        super().__init__()
        self.progressive_blocks = nn.ModuleList([])
        self.rgb_layers         = nn.ModuleList([])
        self.activation         = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            conv_in_chan  = int(in_channels * factors[i])
            conv_out_chan = int(in_channels * factors[i-1])
            self.progressive_blocks.append(Conv2dBlock(conv_in_chan, conv_out_chan, use_pn=False))
            self.rgb_layers.append(WSConv2d(img_channels, conv_in_chan, kernel_size=1, stride=1, padding=0))

        self.out_rgb = WSConv2d(img_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.rgb_layers.append(self.out_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            WSConv2d(in_channels + 1, in_channels),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, in_channels, kernel_size=4, padding=0),
            nn.LeakyReLU(0.2),
            WSConv2d(in_channels, 1, kernel_size=1, padding=0)
        )

    def fade_in(self, alpha, downscaled, out):
        return alpha * out + (1-alpha) * downscaled

    def minibatch_std(self, input):
        batch_statistics = torch.std(input, dim=0).mean().repeat(input.shape[0], 1, input.shape[2], input.shape[3])
        return torch.cat([input, batch_statistics], dim=1)

    def forward(self, input, alpha, steps):
        cur_step = len(self.progressive_blocks) - steps
        out = self.activation(self.rgb_layers[cur_step](input))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)
        
        downscaled = self.activation(self.rgb_layers[cur_step + 1](self.avg_pool(input)))
        out = self.avg_pool(self.progressive_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step+1, len(self.progressive_blocks)):
            out = self.progressive_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)
    

if __name__ == "__main__":
    z_dim = 512
    in_channels = 1024
    gen = Generator(z_dim, in_channels, img_channels=3)
    critic = Critic(in_channels, img_channels=3)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = int(math.log2(img_size / 4))
        x = torch.randn((1, z_dim, 1, 1))
        z = gen(x, 0.5, steps=num_steps)
        assert z.shape == (1, 3, img_size, img_size)
        out = critic(z, alpha=0.5, steps=num_steps)
        assert out.shape == (1, 1)

        print(f"Success at img_size: {img_size}")