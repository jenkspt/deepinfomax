
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import conv_output_size


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, *args, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    """
    DCGAN Discriminator architecture from (add github repo url)

    Except use ReLU instead of LeakyReLu, and batch norm after all hidden layers
    Removed single conv layer with stride since input is 32x32 instead of 64x64
    Single hidden layer w/ 1024 units after last conv layer
    """
    def __init__(self,
            input_shape=(3, 32, 32),
            encoding_size=64,
            feature_layer=3):

        super().__init__()

        self.encoding_size = encoding_size

        in_channels, H, W = input_shape
        out_channels = 64

        lfC, lfH, lfW = input_shape # initial shape of local feature

        layers = []
        for i in range(3):
            layers += [ConvLayer(in_channels, out_channels, 4, 2, 1, bias=False)]
            H, W = conv_output_size((H,W), 4, 2, 1)
            if i+1 == feature_layer:    # Save the shape of the local feature layer
                lfC, lfH, lfW = out_channels, H, W
            in_channels, out_channels = out_channels, out_channels * 2

        self.local_feature_shape = (lfC, lfH, lfW)
        #print(f'C {out_channels//2}, H {H}, W {W}') 
        #print(f'lfC {lfC}, lfH {lfH}, lfW {lfW}') 
        # Local feature encoder
        self.C = nn.Sequential(*layers[:feature_layer])
        self.f0 = nn.Sequential(*layers[feature_layer:])
        self.f1 = nn.Sequential(
                nn.Linear(out_channels//2 * H * W, 1024, bias=False),
                nn.BatchNorm1d(1024),
                nn.ReLU(),
                nn.Linear(1024, self.encoding_size, bias=True)
            )

    def forward(self, x):
        M = self.C(x)                       # Local feature
        h = self.f0(M)
        y = self.f1(h.view(h.size(0), -1))  # Global encoding
        return y, M



class GlobalDIM(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        
        self.sequential = nn.Sequential(
                nn.Linear(in_features, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )

    def forward(self, x):
        return self.sequential(x)


class LocalDIM1(nn.Module):
    """ Concat and Convolve Architecture """
    def __init__(self, in_channels):
        super().__init__()
        self.sequential = nn.Sequential(
                nn.Conv2d(in_channels, 512, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(512, 1, kernel_size=1)
            )

    def forward(self, x):
        return self.sequential(x)

class LocalDIM2(nn.Module):
    """ Encode and Dot Architecture """
    def __init__(self, encoding_size):
        super().__init__()

        self.G1 = nn.Sequential(
                nn.Linear(encoding_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, 2048)
            )
        self.G2 = nn.Sequential(
                nn.Linear(encoding_size, 2048),
                nn.ReLU()
            )

        self.L1 = nn.Sequential(
                nn.Conv2d(1, 2048, 1),
                nn.ReLU(),
                nn.Conv2d(2048, 2048, 1)
            )

        self.L2 = nn.Sequential(
                nn.Conv2d(1, 2048, 1),
                nn.ReLU(),
            )

    def forward(self, y, M):
        raise NotImplementedError
        g = self.G1(y) + self.G2(y)
        l = self.L1(M) + self.L2(M)



class PriorMatch(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.sequential = nn.Sequential(
                nn.Linear(in_features, 1000),
                nn.ReLU(),
                nn.Linear(1000, 200),
                nn.ReLU(),
                nn.Linear(200, 1)
            )

    def forward(self, x):
        return self.sequential(x)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)

    # Encoder
    x = torch.randn([4,3,32,32])

    encoder = Encoder(feature_layer=2)
    y, M = encoder(x)

    # Global
    global_input = torch.randn(4, 512*28**2+64) 
    global_disc = GlobalDIM(in_features=512*28**2+64)
    logits = global_disc(global_input)

    # Local 
    local_input = torch.randn(4, 512+64, 28, 28)
    local_disc = LocalDIM(in_channels=512+64)
    logits = local_disc(local_input)
    print(logits.shape)

    # Prior
    z = torch.rand(64).to(device)
    prior_disc = PriorMatch(in_features=64)
    prior_disc.to(device)
    prior_disc(z)
