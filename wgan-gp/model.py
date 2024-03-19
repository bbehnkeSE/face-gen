from torch import nn

class Generator(nn.Module):
    """
    WGAN-GP Generator
    Params:
        z_dim:        Dimension of the noise vector used to kick off generation
        hidden_dim:   Dimension of the hidden layers
        img_channels: Number of channels in the images (Black-and-white images only have one, RBG have three)
    """
    def __init__(self, z_dim, hidden_dim, img_channels):
        super(Generator, self).__init__()
        self.z_dim        = z_dim
        self.hidden_dim   = hidden_dim
        self.img_channels = img_channels

        self.gen = nn.Sequential(
            self._gen_block(z_dim,         hidden_dim*8, kernel_size=4, stride=1, padding=0),
            self._gen_block(hidden_dim*8,  hidden_dim*4, kernel_size=4, stride=2, padding=1),
            self._gen_block(hidden_dim*4,  hidden_dim*2, kernel_size=4, stride=2, padding=1),
            self._gen_block(hidden_dim*2,  hidden_dim,   kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose2d(hidden_dim, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)
    
    def _gen_block(self, input_channels, output_channels, kernel_size, stride, padding):
        """
        Build sequence of conv transpose layers that make up a single block of the generator
        Params:
            input_channels:  Number of channels for the input features
            output_channels: Number of desired channels for the output
            kernel_size:     Size of the filter used for the convolution (kernel_size, kernel_size)
            stride:          Size of the steps taken by the filter over the input
            padding:         Size of the border to place around the image
        Returns:
            A sequential object
        """
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels, 
                output_channels, 
                kernel_size, 
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU()
        )
    

class Critic(nn.Module):
    """
    WGAN-GP Critic
    Params:
        hidden_dim:   Dimension of the hidden layers
        img_channels: Number of channels in the images (Black-and-white images only have one, RBG have three)
    """
    def __init__(self, hidden_dim, img_channels):
        super(Critic, self).__init__()
        self.hidden_dim   = hidden_dim
        self.img_channels = img_channels

        self.crit = nn.Sequential(
            nn.Conv2d(img_channels, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            self._crit_block(hidden_dim,   hidden_dim*2, kernel_size=4, stride=2, padding=1),
            self._crit_block(hidden_dim*2, hidden_dim*4, kernel_size=4, stride=2, padding=1),
            self._crit_block(hidden_dim*4, hidden_dim*8, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(hidden_dim*8,        1,            kernel_size=4, stride=1, padding=0)
        )

    def forward(self, x):
        return self.crit(x)

    def _crit_block(self, input_channels, output_channels, kernel_size, stride, padding):
        """
        Build sequence of conv layers that make up a single block of the discriminator
        Params:
            Params:
            input_channels:  Number of channels for the input features
            output_channels: Number of desired channels for the output
            kernel_size:     Size of the filter used for the convolution (kernel_size, kernel_size)
            stride:          Size of the steps taken by the filter over the input
            padding:         Size of the border to place around the image
        Returns:
            A sequential object
        """
        return nn.Sequential(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(0.2)
        )
    

def weights_init(model):
    """
    Helper function to initialize the weights of the GAN with
    a mean of 0 and a std deviation of 0.02
    Taken from "https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    Params:
        model: The model whose weights need to be initialized
    """
    classname = model.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.normal_(model.bias.data, 0)