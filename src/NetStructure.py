import torch
import torch.nn as nn


class Conv(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(C_out, C_out, 3, 1, 1),
            nn.ReLU(True)

            # new version
            # nn.Conv2d(C_in, C_out, 3, 1, 1),
            # nn.BatchNorm2d(C_out),
            #     # 防止过拟合
            # nn.Dropout(0.3),
            # nn.LeakyReLU(),
            #
            # nn.Conv2d(C_out, C_out, 3, 1, 1),
            # nn.BatchNorm2d(C_out),
            #     # 防止过拟合
            # nn.Dropout(0.4),
            # nn.LeakyReLU(),
        )

    def forward(self, item):
        return self.layer(item)

class Conv31(nn.Module):
    def __init__(self, C_in, C_out):
        super(Conv31, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(C_in, C_out, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(C_out, C_out, 1, 1, 0),
            nn.ReLU(True)
        )

    def forward(self, item):
        return self.layer(item)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x


class DownSampling(nn.Module):
    def __init__(self):
        super(DownSampling, self).__init__()
        self.layer = nn.Sequential(
            nn.MaxPool2d(2, 2)
            # nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, item):
        return self.layer(item)

class DownSampling_avg(nn.Module):
    def __init__(self):
        super(DownSampling_avg, self).__init__()
        self.layer = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        )

    def forward(self, item):
        return self.layer(item)


class UpSampling(nn.Module):
    def __init__(self, C_in, C_out):
        super(UpSampling, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, 2, 2, 0)
        )

    def forward(self, item):
        return self.layer(item)

class UpBilinear(nn.Module):
    def __init__(self):
        super(UpBilinear, self).__init__()
        self.layer = nn.Sequential(
            nn.Upsample(size=None, scale_factor=2, mode="bilinear", align_corners=True)
        )

    def forward(self, item):
        return self.layer(item)


