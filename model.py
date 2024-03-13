import torch
import torch.nn as nn
import torch.nn.functional as F

class MappingNetwork(nn.Module):
    """
    Neural network to map input from z-space to w-space
    before AdaIN.

    Input:
        z_dim: Dimension of the input noise vector for the first layer
        h_dim: Dimension of the hidden layers
        w_dim: Dimension of the output intermediate noise vector
    """
    def __init__(self, z_dim, h_dim, w_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, w_dim),
        )

    def forward(self, noise):
        """
        Parameters:
            noise: Tensor of shape (n_samples, z_dim)

        Returns:
            Intermediate noise tensor
        """
        return self.network(noise)
    

class Noise(nn.Module):
    """
    Class to add random noise before AdaIN block

    Input:
        img_channels: Number of channels in the image
    """
    def __init__(self, img_channels):
        super().__init__()
        # Initialize the weights for the channels
        # nn.Parameter allows the weights to be optimized
        self.weight = nn.Parameter(torch.randn(1, img_channels, 1, 1))

    def forward(self, img):
        """
        Adds noise to an image along with trainable weights
        Parameters:
            img: Tensor of shape (n_samples, channels, width, height)

        Returns:
            Images with noise applied
        """
        noise = torch.randn(
            (img.shape[0], 1, img.shape[2], img.shape[3]),
            device=img.device
        )

        return (self.weight * noise) + img
    

class AdaIN(nn.Module):
    """
    Adaptive Instance Normalization

    Parameters:
        channels: Number of channels in image
        w_dim:    Dimension of the itermediate noise
    """
    def __init__(self, channels, w_dim):
        super().__init__()

        # Normalize input
        self.instance_norm = nn.InstanceNorm2d(channels)

        # Split "w" into style and shift transforms (y_s and y_b)
        self.y_scale = nn.Linear(w_dim, channels)
        self.y_shift = nn.Linear(w_dim, channels)

    def forward(self, img, w):
        """
        Parameters:
            img: Tensor of shape (n_samples, channels, width, height)
            w:   Intermediate noise vector
        
        Returns:
            Image after normalization and transformation
        """
        img_norm = self.instance_norm(img)
        y_scale = self.y_scale(w)[:, :, None, None]
        y_shift = self.y_shift(w)[:, :, None, None]

        return (y_scale * img_norm) + y_shift
    

class Synthesis_Net_Block(nn.Module):
    """
    StyleGAN Synthesis_Net Block
    Parameters:
        in_channel:    Dim of the const input (usually w_dim)
        out_channel:   Number of channels in the output
        w_dim:         The dimension of the intermediate noise vector
        kernel_size:   Size of the kernel used in convolutions
        starting_size: The size of the starting image
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 w_dim,
                 kernel_size,
                 starting_size,
                 use_upsample=True):
        super().__init__()
        self.use_upsample = use_upsample

        if use_upsample:
            self.upsample = nn.Upsample((starting_size, starting_size), mode="bilinear")

        self.conv         = nn.Conv2d(in_channel, out_channel, kernel_size, padding=1)
        self.noise        = Noise(out_channel)
        self.adain        = AdaIN(out_channel, w_dim)
        self.activation   = nn.LeakyReLU(.2)

    def forward(self, x, w):
        """
        Parameters:
            x: Synthesis_Net input, feature map of shape (n_samples, channels, width, height)
            w: The intermediate vector
        """
        if self.use_upsample:
            x = self.upsample(x)
        
        x = self.conv(x)
        x = self.noise(x)
        x = self.adain(x, w)
        x = self.activation(x)

        return x
    

class Synthesis_Net(nn.Module):
    """
    Parameters:
        z_dim:          Dim of the noise vector
        mh_dim:         Inner dim of the mapping
        w_dim:          Dim of the intermediate vector
        in_channel:     Dim of the const input (usually w_dim)
        out_channel:    Number of output channels
        kernel_size:    Size of the kernel for convolutions
        hidden_channel: Inner dim
        num_layers:     Number of layers in the synth net
    """
    def __init__(
            self,
            z_dim,
            mh_dim,
            w_dim,
            in_channel,
            out_channel,
            kernel_size,
            hidden_channel,
            starting_size=4,
            num_layers=9,
            alpha=0.2):
        super().__init__()

        self.map = MappingNetwork(z_dim, mh_dim, w_dim)
        self.starting_const = nn.Parameter(torch.ones(1, in_channel, 4, 4))
        self.layers = []
        layer0 = Synthesis_Net_Block(in_channel, hidden_channel, w_dim, kernel_size, starting_size, use_upsample=False)
        self.layers.append(layer0)

        img_size_mult = 2
        for l in range(num_layers - 1):
            self.layers.append(Synthesis_Net_Block(in_channel, hidden_channel, w_dim, kernel_size, starting_size * img_size_mult))
            img_size_mult *= 2