import torch
import torch.nn as nn
import numpy as np


# class CustomMaxPooling2D(nn.Module):
#     """"
#     This class performs 2D max-pooling in a way that the shape of the tensor after max-pooling equals the shape of
#     the input tensor. We achieve this by performing some padding before performing max-pooling. This function is
#     inspired by the Keras implementation of max-pooling. We also use elements of the GitHub repository of the max_pool
#     function in keras/backend/torch/nn.
#
#     Args:
#         kernel_size:    a tuple that indicates the size of the max-pooling kernel (over the spatial dimensions).
#         strides:        a tuple indicating the stride used for each dimension
#
#     """
#
#     def __init__(self, kernel_size, strides=None):
#         super(CustomMaxPooling2D, self).__init__()
#
#         self.kernel_size = kernel_size
#         self.strides = kernel_size if strides is None else strides
#
#     def forward(self, inputs):
#         """"
#         This function performs the custom 2D max-pooling.
#
#         Args:
#             inputs:     a torch tensor of shape 'batch_size x num_channels x ... x ......' where the last dimensions
#                         form the spatial dimensions. E.g. in case of 2D images we have a tensor of size
#                         'batch_size x num_channels x image_width x image_height'
#
#         Returns:
#             a torch tensor with size inputs.size()
#
#         """
#
#         # First we do some padding such that the output shape after max pooling equals the input shape.
#
#         # Grab the number of elements each spatial dimension of the input has and calculate the number of spatial dimensions
#         # that we have
#         spatial_shape = inputs.shape[2:]
#         num_spatial_dims = len(spatial_shape)
#
#         # Initialize the tuple where entries (2*i, 2*i+1) indicate with how many elements to pad the 'left' hand-side of
#         # spatial dimension i and with how many elements to pad the 'right' hand-side of spatial dimension i.
#         padding = ()
#
#         # For every spatial dimension, do ...
#         for i in range(num_spatial_dims):
#
#             # Grab the kernel length, the stride, and the number of elements of the i-th spatial dimension.
#             kernel_length = self.kernel_size[i]
#             stride = self.strides[i]
#             num_elements = spatial_shape[i]
#
#             # Determine how much to pad the 'left' and 'right' side of the i-th spatial dimension. You do this by first
#             # calculating the total amount of padding needed for both 'left' and 'right' side and then dividing this amount
#             # of padding (approximately) equally on both sides.
#             if (num_elements - 1) % stride == 0:
#                 total_padding_length = (kernel_length - 1)
#             else:
#                 total_padding_length = ((kernel_length - 1) - (num_elements - 1) % stride)
#             left_padding = int(np.floor(total_padding_length / 2))
#             right_padding = int(np.ceil(total_padding_length / 2))
#
#             # Determine the required padding for the i-th spatial dimension.
#             padding_size = (left_padding, right_padding)
#
#             # Add this required padding to the 'padding' variable required to perform the padding with nn.functional.pad
#             padding = padding_size + padding
#
#         # Perform the padding
#         inputs = nn.functional.pad(inputs, padding, mode="replicate")
#
#         # Perform the max-pooling
#         return nn.functional.max_pool2d(inputs, kernel_size=self.kernel_size, stride=self.strides)


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
        self.conv = nn.Conv2d(in_channels, out_channels, conv_kernel_size, padding='same')
        self.maxpool = nn.MaxPool2d(maxpool_kernel_size)

        #self.maxpool = CustomMaxPooling2D(maxpool_kernel_size)

    def forward(self, x):

        # First apply the convolutional layer
        y = self.conv(x)

        # Apply the activation function
        y = self.act_func(y)

        # Finally return the value after applying maxpooling
        return self.maxpool(y)


class ConvTransposeBlock(nn.Module):
    """"
    This is a convolution block consisting of an upsampling layer, followed by a transposed convolution layer, and
    finally an activation function

    Args:
        in_channels:            the number of input channels.
        out_channels:           the number of output channels.
        conv_kernel_size:       a tuple (i, j) with the size of the kernel used in the transposed convolutional layer.
                                DEFAULT: (3, 3)
        scale_factor:           an integer defining by how much we multiply the current resolution of the 'image'. So
                                if the current resolution is 10 x 10 and scale_factor=2, then the new resolution is
                                20 x 20. DEFAULT: 2.
        act_func:               the activation function used. DEFAULT: torch.relu

    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=(3, 3), scale_factor=2,
                 act_func=torch.relu):
        super(ConvTransposeBlock, self).__init__()

        # Save the values of the ConvBlock options
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_kernel_size = conv_kernel_size
        self.scale_factor = scale_factor
        self.act_func = act_func

        # Define the upsampling layer and the transposed convolution layer
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.convtranspose = nn.ConvTranspose2d(in_channels, out_channels, conv_kernel_size, padding=1)

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
        self.linear1 = nn.Linear(10 * 10 * 64, 4 * self.latent_dim + 8)
        self.linear2 = nn.Linear(4 * self.latent_dim + 8, self.latent_dim)

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
        self.decoder1 = nn.Sequential(ConvTransposeBlock(64, 32, (3, 3), 2, torch.relu),
                                       ConvTransposeBlock(32, 16, (3, 3), 2, torch.relu),
                                       ConvTransposeBlock(16, 1, (3, 3), 2, torch.sigmoid))

        self.decoder2 = nn.Sequential(ConvTransposeBlock(64, 32, (3, 3), 2, torch.relu),
                                      ConvTransposeBlock(32, 16, (3, 3), 2, torch.relu),
                                      ConvTransposeBlock(16, 1, (3, 3), 2, torch.sigmoid))

        self.decoder3 = nn.Sequential(ConvTransposeBlock(64, 32, (3, 3), 2, torch.relu),
                                      ConvTransposeBlock(32, 16, (3, 3), 2, torch.relu),
                                      ConvTransposeBlock(16, 1, (3, 3), 2, torch.sigmoid))

    def forward(self, x):

        # Apply the first linear layer to create a higher dimensional starting vector that can be supplied to the first
        # convolutional layer after reshaping.
        y = torch.reshape(self.linear1(x), (x.size(0), 64, 10, 10))

        # Apply a relu activation function
        y = torch.relu(y)

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
        latent_dim:         the dimension of the latent codes
        number_of_classes:  the number of classes to predict

    """

    def __init__(self, latent_dim, number_of_classes):
        super(Classifier, self).__init__()
        self.latent_dim = latent_dim
        self.number_of_classes = number_of_classes

        # Define the two linear layers used in the classifier
        self.linear1 = nn.Linear(self.latent_dim, number_of_classes)
        self.linear2 = nn.Linear(number_of_classes, number_of_classes)

    def forward(self, x):
        # Apply the first linear layer, a relu function, the second linear layer, and then a softmax
        return torch.softmax(self.linear2(torch.relu(self.linear1(x))), dim=-1)
