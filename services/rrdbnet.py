"""RRDBNet x4 architecture compatible with official RealESRGAN_x4plus weights.

Torch remains optional and lazy. Key names match the published checkpoint;
no BasicSR package initialization or torchvision training dependencies are needed.
"""


def rrdbnet(num_feat=64, num_block=23, num_grow_ch=32):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class DenseBlock(nn.Module):
        def __init__(self):
            super().__init__()
            for index in range(1, 6):
                channels = num_feat if index == 5 else num_grow_ch
                setattr(self, f'conv{index}', nn.Conv2d(num_feat+(index-1)*num_grow_ch, channels, 3, 1, 1))
            self.lrelu = nn.LeakyReLU(.2, inplace=True)

        def forward(self, x):
            features = [x]
            for index in range(1, 5):
                features.append(self.lrelu(getattr(self, f'conv{index}')(torch.cat(features, 1))))
            return x + .2*self.conv5(torch.cat(features, 1))

    class RRDB(nn.Module):
        def __init__(self):
            super().__init__()
            self.rdb1, self.rdb2, self.rdb3 = DenseBlock(), DenseBlock(), DenseBlock()

        def forward(self, x):
            return x + .2*self.rdb3(self.rdb2(self.rdb1(x)))

    class RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*(RRDB() for _ in range(num_block)))
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, 3, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(.2, inplace=True)

        def forward(self, x):
            feature = self.conv_first(x)
            feature = feature + self.conv_body(self.body(feature))
            feature = self.lrelu(self.conv_up1(F.interpolate(feature, scale_factor=2, mode='nearest')))
            feature = self.lrelu(self.conv_up2(F.interpolate(feature, scale_factor=2, mode='nearest')))
            return self.conv_last(self.lrelu(self.conv_hr(feature)))

    return RRDBNet()
