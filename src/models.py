import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """"
    This is a convolution block consisting of a convolution layer, followed by a maxpooling layer, and finally an
    activation function

    Args:
        in_channels:            the number of input channels.
        out_channels:           the number of output channels.
        conv_kernel_size:       a tuple (i, j) with the size of the kernel used in the convolutional layer.
                                DEFAULT: (3, 3)
        maxpool_kernel_size:    a tuple (i, j) with the size of the kernel used for maxpooling. DEFAULT: (2, 2)
        act_func:               the activation function used. DEFAULT: torch.relu

    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=(3, 3), maxpool_kernel_size=(2, 2),
                 act_func=torch.relu):
        super(ConvBlock, self).__init__()

        # Save the values of the ConvBlock options
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.maxpool_kernel_size = maxpool_kernel_size
        self.act_func = act_func

        # Define the convolutional layer and the maxpooling layer
        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel_size)
        self.maxpool = nn.MaxPool2d(maxpool_kernel_size)

    def forward(self, x):

        # First apply the convolutional layer
        y = self.conv(x)

        # Then apply maxpooling
        y = self.maxpool(y)

        # Finally return the value after having applied the activation function
        return self.act_func(y)


class ConvTransposeBlock(nn.Module):
    """"
    This is a convolution block consisting of an upsampling layer, followed by a transposed convolution layer, and
    finally an activation function

    Args:
        in_channels:            the number of input channels.
        out_channels:           the number of output channels.
        conv_kernel_size:       a tuple (i, j) with the size of the kernel used in the transposed convolutional layer.
                                DEFAULT: (3, 3)
        upsampling_kernel_size: a tuple (i, j) with the size of the kernel used for upsampling. DEFAULT: (2, 2)
        act_func:               the activation function used. DEFAULT: torch.relu

    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=(3, 3), upsampling_kernel_size=(2, 2),
                 act_func=torch.relu):
        super(ConvTransposeBlock, self).__init__()

        # Save the values of the ConvBlock options
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.upsampling_kernel_size = upsampling_kernel_size
        self.act_func = act_func

        # Define the upsampling layer and the transposed convolution layer
        self.upsample = nn.Upsample(upsampling_kernel_size)
        self.convtranspose = nn.ConvTranspose2d(in_channels, out_channels, conv_kernel_size)

    def forward(self, x):
        # First apply the upsampling
        y = self.upsample(x)

        # Then apply the transposed convolution
        y = self.convtranspose(y)

        # Finally return the value after having applied the activation function
        return self.act_func(y)


class Encoder(nn.Module):
    """"
    This is the neural network that maps the images to the latent codes.

    Args:
        latent_dim:     the dimension of the (Euclidean) latent code/vector that we use
    """

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        # Define 3 convolution blocks
        self.convblock1 = ConvBlock(3, 16, (3, 3), (2, 2), torch.relu)
        self.convblock2 = ConvBlock(16, 32, (3, 3), (2, 2), torch.relu)
        self.convblock3 = ConvBlock(32, 64, (3, 3), (2, 2), torch.relu)

        # Define two linear layers that make sure that we go to the right latent dimension
        self.linear1 = nn.Linear(10 * 10 * 64, 208)
        self.linear2 = nn.Linear(208, self.latent_dim)

    def forward(self, x):

        # Apply the 3 convolution blocks
        y = self.convblock1(x)
        y = self.convblock2(y)
        y = self.convblock3(y)

        # Flatten the output of the convolution blocks. NOTE: we do want to keep the batch dimension!
        y = torch.flatten(y, start_dim=1)

        # Apply the first linear layer and a relu activation function to the flattened output
        y = self.linear1(y)
        y = torch.relu(y)

        # Yield the final output by applying one last linear layer
        return self.linear2(y)


class Decoder(nn.Module):
    """"
    This is the neural network that maps the latent codes back to the images

    Args:
        latent_dim:     the dimension of the (Euclidean) latent code/vector that we use
    """

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim

        # Define the linear layer that transforms the latent code into a higher dimensional vector
        self.linear1 = nn.Linear(self.latent_dim, 10 * 10 * 64)

        # Define 3 decoders composed of three transposed convolution blocks. Each decoder decodes one fluorescent
        # channel
        self.decoder1 = nn.Sequential(ConvTransposeBlock(64, 32, (3, 3), (2, 2), torch.relu),
                                       ConvTransposeBlock(32, 16, (3, 3), (2, 2), torch.relu),
                                       ConvTransposeBlock(16, 1, (3, 3), (2, 2), torch.sigmoid))

        self.decoder2 = nn.Sequential(ConvTransposeBlock(64, 32, (3, 3), (2, 2), torch.relu),
                                      ConvTransposeBlock(32, 16, (3, 3), (2, 2), torch.relu),
                                      ConvTransposeBlock(16, 1, (3, 3), (2, 2), torch.sigmoid))

        self.decoder3 = nn.Sequential(ConvTransposeBlock(64, 32, (3, 3), (2, 2), torch.relu),
                                      ConvTransposeBlock(32, 16, (3, 3), (2, 2), torch.relu),
                                      ConvTransposeBlock(16, 1, (3, 3), (2, 2), torch.sigmoid))

    def forward(self, x):

        # Apply the first linear layer to create a higher dimensional starting vector that can be supplied to the first
        # convolutional layer after reshaping.
        y = torch.reshape(self.linear1(x), (x.shape(0), 64, 10, 10))

        # Apply the decoders to get each fluorescence channels
        x1 = self.decoder1(y)
        x2 = self.decoder2(y)
        x3 = self.decoder3(y)

        # Return the concatenated result
        return torch.concatenate([x1, x2, x3], dim=1)


class Classifier(nn.Module):
    """"
    This defines the classifier that is used to classify the images based on their latent codes.

    Args:
        latent_dim:     the dimension of the latent codes

    """

    def __init__(self, latent_dim):
        super(Classifier, self).__init__()
        self.latent_dim = latent_dim

        # Define the two linear layers used in the classifier
        self.linear1 = nn.Linear(self.latent_dim, 6)
        self.linear2 = nn.Linear(6, 6)

    def forward(self, x):
        # Apply the first linear layer, a relu function, the second linear layer, and then a softmax
        return torch.softmax(self.linear2(torch.relu(self.linear1(x))), dim=-1)
