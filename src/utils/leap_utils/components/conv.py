import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class KiTTiConvDecoder(nn.Module):
    """Convolutional decoder for KittiMask"""
    def __init__(self, z_dim=10, nc=3, hidden_dim=256):
        super().__init__()
        self.upsample = nn.Sequential(
                                        nn.Linear(z_dim, hidden_dim),               # B, hidden_dim
                                        View((-1, hidden_dim, 1, 1)),               # B, hidden_dim,  1,  1
                                        nn.ReLU(True),
                                        nn.ConvTranspose2d(hidden_dim, 64, 4),      # B,  64,  4,  4
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True),
                                        nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True),
                                        nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(True),
                                        nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(True),
                                        nn.ConvTranspose2d(32, nc, 4, 2, 1)  # B, nc, 64, 64
                                     ) 
    def forward(self, x):
        return self.upsample(x)

class KiTTiConvEncoder(nn.Module):
    """Convolutional encoder for KittiMask"""
    def __init__(self, z_dim=10, nc=3, hidden_dim=256):
        super().__init__()
        self.downsample = nn.Sequential(
                                        nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, hidden_dim, 4, 1),     # B, hidden_dim,  1,  1
                                        nn.BatchNorm2d(hidden_dim),
                                        nn.ReLU(True),
                                        View((-1, hidden_dim*1*1)),       # B, hidden_dim
                                        nn.Linear(hidden_dim, z_dim)             # B, z_dim
                                        )
    def forward(self, x):
        return self.downsample(x)

class BallConvEncoder(nn.Module):
    """Convolutional encoder for Ball Dataset"""
    def __init__(self, z_dim=10, nc=3, nf=16):
        super().__init__()
        sequence = [
            # input is (ni) x 64 x 64
            nn.Conv2d(nc, nf, 7, 1, 3),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
            View((-1, nf * 4 * 16 * 16)),
            nn.Linear(nf * 4 * 16 * 16, z_dim)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, feat):
        return self.model(feat)

class BallConvDecoder(nn.Module):
    """Convolutional decoder for Ball dataset"""
    def __init__(self, z_dim=10, nc=3, nf=16):
        super().__init__()
        sequence = [
            # input is (nf * 4) x 16 x 16
            nn.Linear(z_dim, nf * 4 * 16 * 16),
            View((-1, nf * 4,  16, 16)),
            nn.ConvTranspose2d(nf * 4, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.Conv2d(nf * 4, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 4) x 32 x 32
            nn.ConvTranspose2d(nf * 2, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # input is (nf * 2) x 64 x 64
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, feat):
        return self.model(feat)

class SyntheticConvEncoder(nn.Module):
    """Convolutional encoder for Ball Dataset"""
    def __init__(self, z_dim=10, nc=3, nf=16, h=64, w=64):
        super().__init__()
        norm_layer = 'Batch'
        self.linear_in_features = None
        self.z_dim = z_dim
        self.h = h
        self.w = w
        sequence = [
            # input is (ni) x 64 x 64
            nn.Conv2d(nc, nf, 7, 1, 3),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
            # View((-1, nf * 4 * 16 * 16)),
            nn.Flatten(),
            nn.Linear(nf * 4 * (self.h//(2**2)) * (self.w//(2**2)), z_dim)
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, feat):
        # x = self.model(feat)
        # if self.linear_in_features is None:
        #     self.linear_in_features = x.shape[1]
        #     self.classifier = nn.Linear(self.linear_in_features, self.z_dim).cuda()
        return self.model(feat)

class SyntheticConvDecoder(nn.Module):
    """Convolutional decoder for Ball dataset"""
    def __init__(self, z_dim=10, nc=3, nf=16, h=64, w=64):
        super().__init__()
        norm_layer = 'Batch'    
        self.h = h
        self.w = w
        self.nc = nc
        self.nf = nf
        
        # Start size calculation: Choose a start size that will be scaled up correctly
        start_h = h // 4
        start_w = w // 4

        # Calculate output padding
        out_pad_h = (h - start_h * 4) % 2
        out_pad_w = (w - start_w * 4) % 2

        # Define the sequence of layers
        sequence = [
            # Input is z_dim (flat vector)
            nn.Linear(z_dim, nf * 4 * start_h * start_w),
            nn.Unflatten(1, (nf * 4, start_h, start_w)),
            # First upsampling
            nn.ConvTranspose2d(nf * 4, nf * 4, 4, 2, 1, output_padding=(1 if h % 4 != 0 else 0, 1 if w % 4 != 0 else 0)),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Second upsampling
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, output_padding=(1 if h % 2 != 0 else 0, 1 if w % 2 != 0 else 0)),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Final convolution to adjust channel size to nf * 2
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            View((-1, nf*2)),
            nn.Linear(nf *2, self.nc)
        ]

        self.model = nn.Sequential(*sequence)


    def forward(self, feat):
        output = self.model(feat)
        return output

class SyntheticMultiConvEncoder(nn.Module):
    """Convolutional encoder for Ball Dataset"""
    def __init__(self, z_dim=10, nc=2, nf=16, h=64, w=64):
        super().__init__()
        norm_layer = 'Batch'
        self.linear_in_features = None
        self.z_dim = z_dim
        self.h = h
        self.w = w
        sequence = [
            # input is (ni) x 64 x 64
            nn.Conv2d(nc, nf, 7, 1, 3),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf, 5, 1, 2),
            nn.BatchNorm2d(nf) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf) x 64 x 64
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 2, 3, 1, 1),
            nn.BatchNorm2d(nf * 2) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 2) x 32 x 32
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4) if norm_layer == 'Batch' else nn.InstanceNorm2d(nf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # feat size (nf * 4) x 16 x 16
            # View((-1, nf * 4 * 16 * 16)),
            nn.Flatten(),
            nn.Linear(nf * 4 * (self.h//(2**2)) * (self.w//(2**2)), z_dim)
        ]

        self.model = nn.Sequential(*sequence)