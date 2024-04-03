import torch
from   torch import nn

FACTORS = [1., 1., 1., 1., 0.5, 0.25, 0.125, 0.0625, 0.03125]

class WtScaleConv2d(nn.Module):
    """
    Provides a weight-scaled 2D convolutional layer, which can be useful for improving training stability
    and convergence in neural networks. 
    The bias term is handled separately to avoid redundancy during computation.
    """
    def __init__(self, in_chan, out_chan, kernel_size=3, stride=1, padding=1, gain=2):
        super().__init__()
        self.conv      = nn.Conv2d(in_chan, out_chan, kernel_size, stride, padding)
        self.scale     = (gain / (in_chan * kernel_size ** 2)) ** 0.5
        self.bias      = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)


    def forward(self, x):
        return self.conv(x) * self.scale + self.bias.view(1, self.bias.shape[0], 1, 1)
    

class WtScaleLinear(nn.Module):
    """
    Provides a linear layer with weight scaling, which can be useful for improving training stability
    and convergence in neural networks.
    The bias term is handled separately to avoid redundancy during computation.
    """
    def __init__(self, in_features, out_features, gain=2):
        super().__init__()
        self.linear      = nn.Linear(in_features, out_features)
        self.scale       = (gain / in_features) ** 0.5
        self.bias        = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)


    def forward(self, x):
        return self.linear(x) * self.scale + self.bias


class MappingLayers(nn.Module):
    """
    Takes a noise vector 'z' and maps to an intermediate
    noise vector 'w.'
    """
    def __init__(self, z_dim,w_dim):
        super().__init__()
        self.map = nn.Sequential(
            WtScaleLinear(z_dim, w_dim),
            nn.ReLU(),
            WtScaleLinear(w_dim, w_dim),
            nn.ReLU(),
            WtScaleLinear(w_dim, w_dim),
            nn.ReLU(),
            WtScaleLinear(w_dim, w_dim),
            nn.ReLU(),
            WtScaleLinear(w_dim, w_dim),
            nn.ReLU(),
            WtScaleLinear(w_dim, w_dim),
            nn.ReLU(),
            WtScaleLinear(w_dim, w_dim),
            nn.ReLU(),
            WtScaleLinear(w_dim, w_dim),
        )


    def forward(self, noise):
        return self.map(noise)
        

class InjectNoise(nn.Module):
    """
    Creates random noise injection for the AdaIN blocks.
    Creates noise tensor of shape (1, img_height, img_width), multiplies
    that tensor by learned (img_chan) values to create a tensor of
    shape (img_chan, img_width, img_height), then adds the noise to the image
    """
    def __init__(self, img_chan):
        super().__init__()
        # Learned values to multiply with noise
        self.weight = nn.Parameter(torch.randn(1, img_chan, 1, 1))


    def forward(self, x):
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)

        return x + (noise * self.weight)
    

class AdaIN(nn.Module):
    """
    Takes the instance normalization of an image and applies
    style scale (y_s), and style bias (y_b), which is learned from the intermediate
    noise vector 'w.'
    """
    def __init__(self, img_chan, w_dim):
        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(img_chan)
        self.style_scale   = WtScaleLinear(w_dim, img_chan)
        self.style_bias    = WtScaleLinear(w_dim, img_chan)


    def forward(self, x, w):
        x                  = self.instance_norm(x)
        y_s                = self.style_scale(w)[:, :, None, None]
        y_b                = self.style_bias(w)[:, :, None, None]

        return y_s * x + y_b
    

class GenBlock(nn.Module):
    def __init__(self, in_chan, out_chan, w_dim):
        super().__init__()
        self.conv1 = WtScaleConv2d(in_chan, out_chan)
        self.conv2 = WtScaleConv2d(out_chan, out_chan)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.inject_noise1 = InjectNoise(out_chan)
        self.inject_noise2 = InjectNoise(out_chan)

        self.adain1 = AdaIN(out_chan, w_dim)
        self.adain2 = AdaIN(out_chan, w_dim)


    def forward(self, x, w):
        x = self.adain1(self.activation(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.activation(self.inject_noise2(self.conv2(x))), w)

        return x
    

class Generator(nn.Module):
    def __init__(self, z_dim, w_dim, in_chan, img_chan=3):
        super().__init__()
        self.map               = MappingLayers(z_dim, w_dim)
        self.activation        = nn.LeakyReLU(0.2, inplace=True)

        # Build initial Generator block
        self.i_const           = nn.Parameter(torch.ones((1, in_chan, 4, 4)))
        self.i_adain1          = AdaIN(in_chan, w_dim)
        self.i_adain2          = AdaIN(in_chan, w_dim)
        self.i_inject_noise1   = InjectNoise(in_chan)
        self.i_inject_noise2   = InjectNoise(in_chan)
        # self.i_conv            = nn.Conv2d(in_chan, in_chan, kernel_size=3, stride=1, padding=1)
        self.i_conv            = WtScaleConv2d(in_chan, in_chan)  ### Maybe replace with nn.Conv2d ###
        self.i_rgb             = WtScaleConv2d(in_chan, img_chan, kernel_size=1, padding=0)

        self.progessive_blocks = nn.ModuleList([])
        self.rgb_layers        = nn.ModuleList([self.i_rgb])

        # Build rest of the Generator blocks
        for i in range(len(FACTORS) - 1):
            h_in_chan          = int(in_chan * FACTORS[i])
            h_out_chan         = int(in_chan * FACTORS[i + 1])

            self.progessive_blocks.append(GenBlock(h_in_chan, h_out_chan, w_dim))
            self.rgb_layers.append(WtScaleConv2d(h_out_chan, img_chan, kernel_size=1, padding=0))


    def forward(self, noise, alpha, steps):
        w = self.map(noise)
        x = self.i_adain1(self.i_inject_noise1(self.i_const), w)
        x = self.i_conv(x)
        x = self.i_adain2(self.activation(self.i_inject_noise2(x)), w)

        if steps == 0:
            return self.i_rgb(x)
        
        for step in range(steps):
            upscaled = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear")
            x        = self.progessive_blocks[step](upscaled, w)

        upscaled     = self.rgb_layers[steps - 1](upscaled)
        x            = self.rgb_layers[steps](x)

        return self.fade_in(upscaled, x, alpha)


    def fade_in(self, upscaled, native, alpha):
        return (alpha * native) + ((1 - alpha) * upscaled)
    

class CritBlock(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv1      = WtScaleConv2d(in_chan, out_chan)
        self.conv2      = WtScaleConv2d(out_chan, out_chan)
        self.activation = nn.LeakyReLU(0.2)


    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))

        return x


class Critic(nn.Module):
    def __init__(self, in_chan, img_chan):
        super().__init__()
        self.progressive_blocks = nn.ModuleList([])
        self.rgb_layers         = nn.ModuleList([])
        self.activation         = nn.LeakyReLU(0.2)

        for i in range(len(FACTORS) - 1, 0, -1):
            h_in_chan  = int(in_chan * FACTORS[i])
            h_out_chan = int(in_chan * FACTORS[i - 1])

            self.progressive_blocks.append(CritBlock(h_in_chan, h_out_chan))
            self.rgb_layers.append(WtScaleConv2d(img_chan, h_in_chan, kernel_size=1, padding=0))

        self.rgb_layers.append(WtScaleConv2d(img_chan, in_chan, kernel_size=1, padding=0))
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.f_block = nn.Sequential(
            WtScaleConv2d(in_chan + 1, in_chan),
            nn.LeakyReLU(0.2),
            WtScaleConv2d(in_chan, in_chan, kernel_size=4, padding=0),
            nn.LeakyReLU(0.2),
            WtScaleConv2d(in_chan, 1, kernel_size=1, padding=0)
        )


    def forward(self, x, alpha, steps):
        cur_step = len(self.progressive_blocks) - steps
        from_rgb = self.activation(self.rgb_layers[cur_step](x))

        if steps == 0:
            from_rgb = self.minibatch_std(from_rgb)
            return self.f_block(from_rgb).view(from_rgb.shape[0], -1)
        
        downscaled = self.activation(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        from_rgb   = self.avg_pool(self.progressive_blocks[cur_step](from_rgb))
        from_rgb   = self.fade_in(downscaled, from_rgb, alpha)

        for step in range(cur_step + 1, len(self.progressive_blocks)):
            from_rgb = self.avg_pool(self.progressive_blocks[step](from_rgb))

        return self.f_block(self.minibatch_std(from_rgb)).view(from_rgb.shape[0], -1)


    def fade_in(self, downscaled, native, alpha):
        return (alpha * native) + ((1 - alpha) * downscaled)
    

    def minibatch_std(self, x):
        batch_stats = torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])

        return torch.cat([x, batch_stats], dim=1)