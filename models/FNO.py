import torch
import torch.nn as nn
import torch.fft as fft

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 # Number of Fourier modes to retain for height
        self.modes2 = modes2 # Number of Fourier modes to retain for width (for rfft2, this implies W//2 + 1 modes)

        self.scale = (1 / (in_channels * out_channels))
        # Learnable complex weights for the low-frequency modes
        # Weights for the top-left frequency quadrant
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        # Weights for the bottom-left frequency quadrant (corresponding to negative frequencies in the first spatial dimension)
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

        # Linear layer for skip connection (spatial domain transformation)
        # A 1x1 convolution (Conv2d) is typically used in 2D FNOs for this purpose,
        # acting as a channel-wise linear projection in the spatial domain.
        self.linear_skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        batchsize = x.shape[0]
        height = x.shape[-2]
        width = x.shape[-1]

        # 1. Compute 2D Real Fast Fourier Transform (RFFT)
        # x_ft has shape (batch, in_channels, H, W//2 + 1) for real input
        x_ft = torch.fft.rfft2(x)

        # Create an output tensor to store the filtered Fourier modes
        # It will have the same dimensions as x_ft, but with out_channels
        out_ft = torch.zeros(batchsize, self.out_channels, height, width // 2 + 1, dtype=torch.cfloat, device=x.device)

        # 2. Multiply the lower-frequency modes with the learnable complex weights
        # Apply weights to the top-left and bottom-left frequency quadrants of the RFFT output
        # These slices represent the low-frequency modes to be modified.
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)

        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # 3. Perform the inverse 2D Fast Fourier Transform (IFFT)
        # The 's' argument ensures the output spatial dimensions match the original input
        x_spectral_out = torch.fft.irfft2(out_ft, s=(height, width))

        # 4. Add the result from a linear skip connection (spatial domain transformation)
        x_skip_out = self.linear_skip(x)

        return x_spectral_out + x_skip_out



class FNO2d(nn.Module):
    def __init__(self, in_channels, out_channels, width, modes1, modes2):
        super(FNO2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2

        # Lifting layer: projects input channels to 'width' dimension
        self.lifting = nn.Linear(in_channels, self.width) # Input is (..., C)

        # Spectral Convolution layers
        # Each SpectralConv2d block increases the feature dimension from 'width' to 'width'
        self.spectral_conv1 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.spectral_conv2 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.spectral_conv3 = SpectralConv2d(self.width, self.width, modes1, modes2)
        self.spectral_conv4 = SpectralConv2d(self.width, self.width, modes1, modes2)

        # Linear layers for transformations between spectral blocks
        # These are 1x1 convolutions after channel-first (B, C, H, W) permute
        self.linear1 = nn.Conv1d(self.width, self.width, 1)
        self.linear2 = nn.Conv1d(self.width, self.width, 1)
        self.linear3 = nn.Conv1d(self.width, self.width, 1)

        # Projection layer: maps 'width' dimension back to output channels
        self.projection = nn.Linear(self.width, out_channels) # Output is (..., C)

        # Activation function
        self.activation = nn.GELU()

    def forward(self, x):
        # x shape: (batch_size, in_channels, height, width)

        # 1. Lifting layer
        # Permute to (batch_size, height, width, in_channels) for Linear layer
        x = x.permute(0, 2, 3, 1)
        x = self.lifting(x) # (batch_size, height, width, self.width)
        # Permute back to (batch_size, self.width, height, width) for Conv layers
        x = x.permute(0, 3, 1, 2)

        # 2. Sequence of SpectralConv2d blocks with activation and linear transformations
        x1 = self.spectral_conv1(x)
        x2 = self.linear1(x.view(x.shape[0], x.shape[1], -1)).view(x.shape) # Apply 1x1 conv on flattened H*W, then reshape
        x = self.activation(x1 + x2)

        x1 = self.spectral_conv2(x)
        x2 = self.linear2(x.view(x.shape[0], x.shape[1], -1)).view(x.shape)
        x = self.activation(x1 + x2)

        x1 = self.spectral_conv3(x)
        x2 = self.linear3(x.view(x.shape[0], x.shape[1], -1)).view(x.shape)
        x = self.activation(x1 + x2)

        # 3. Projection layer
        # Permute to (batch_size, height, width, self.width) for Linear layer
        x = x.permute(0, 2, 3, 1)
        x = self.projection(x) # (batch_size, height, width, out_channels)
        # Permute back to (batch_size, out_channels, height, width) for final output consistency
        x = x.permute(0, 3, 1, 2)

        return x
