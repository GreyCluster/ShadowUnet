from NetStructure import *

import torch
import torch.nn as nn


def crop_tensor(source, target_tensor):
    target_size0 = target_tensor.size()[2]
    target_size1 = target_tensor.size()[3]
    tensor_size0 = source.size()[2]
    tensor_size1 = source.size()[3]
    delta0 = tensor_size0 - target_size0
    delta0 = delta0 // 2
    delta1 = tensor_size1 - target_size1
    delta1 = delta1 // 2
    return source[:, :, delta0:tensor_size0 - delta0, delta1:tensor_size1 - delta1]


class Unet_origin(nn.Module):
    def __init__(self):
        super(Unet_origin, self).__init__()
        self.conv1 = Conv(4, 64)
        self.down1 = DownSampling()

        self.conv2 = Conv(64, 128)
        self.down2 = DownSampling()

        self.conv3 = Conv(128, 256)
        self.down3 = DownSampling()

        self.conv4 = Conv(256, 512)
        self.down4 = DownSampling()

        self.conv5 = Conv(512, 1024)

        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)

        self.conv6 = Conv(1024, 512)
        self.conv7 = Conv(512, 256)
        self.conv8 = Conv(256, 128)
        self.conv9 = Conv(128, 64)
        self.conv10 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        # level1
        x1 = self.conv1(x)
        x2 = self.down1(x1)

        # level2
        x2 = self.conv2(x2)
        x3 = self.down2(x2)

        # level3
        x3 = self.conv3(x3)
        x4 = self.down3(x3)

        # level4
        x4 = self.conv4(x4)
        x5 = self.down4(x4)

        # level5
        x5 = self.conv5(x5)
        y4 = self.up1(x5)

        # level4
        t4 = crop_tensor(x4, y4)
        y4 = torch.cat([t4, y4], dim=1)
        y4 = self.conv6(y4)
        y3 = self.up2(y4)

        # level3
        t3 = crop_tensor(x3, y3)
        y3 = torch.cat([t3, y3], dim=1)
        y3 = self.conv7(y3)
        y2 = self.up3(y3)

        # level2
        t2 = crop_tensor(x2, y2)
        y2 = torch.cat([t2, y2], dim=1)
        y2 = self.conv8(y2)
        y1 = self.up4(y2)

        # level1
        t1 = crop_tensor(x1, y1)
        y1 = torch.cat([t1, y1], dim=1)
        y1 = self.conv9(y1)

        U_out = self.conv10(y1)

        return U_out


class Unet_avgpool(nn.Module):
    def __init__(self):
        super(Unet_avgpool, self).__init__()
        self.conv1 = Conv(4, 64)
        self.down1 = DownSampling_avg()

        self.conv2 = Conv(64, 128)
        self.down2 = DownSampling_avg()

        self.conv3 = Conv(128, 256)
        self.down3 = DownSampling_avg()

        self.conv4 = Conv(256, 512)
        self.down4 = DownSampling_avg()

        self.conv5 = Conv(512, 1024)

        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)

        self.conv6 = Conv(1024, 512)
        self.conv7 = Conv(512, 256)
        self.conv8 = Conv(256, 128)
        self.conv9 = Conv(128, 64)
        self.conv10 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        # level1
        x1 = self.conv1(x)
        x2 = self.down1(x1)

        # level2
        x2 = self.conv2(x2)
        x3 = self.down2(x2)

        # level3
        x3 = self.conv3(x3)
        x4 = self.down3(x3)

        # level4
        x4 = self.conv4(x4)
        x5 = self.down4(x4)

        # level5
        x5 = self.conv5(x5)
        y4 = self.up1(x5)

        # level4
        t4 = crop_tensor(x4, y4)
        y4 = torch.cat([t4, y4], dim=1)
        y4 = self.conv6(y4)
        y3 = self.up2(y4)

        # level3
        t3 = crop_tensor(x3, y3)
        y3 = torch.cat([t3, y3], dim=1)
        y3 = self.conv7(y3)
        y2 = self.up3(y3)

        # level2
        t2 = crop_tensor(x2, y2)
        y2 = torch.cat([t2, y2], dim=1)
        y2 = self.conv8(y2)
        y1 = self.up4(y2)

        # level1
        t1 = crop_tensor(x1, y1)
        y1 = torch.cat([t1, y1], dim=1)
        y1 = self.conv9(y1)

        U_out = self.conv10(y1)

        return U_out


class Unet_avgpool_reduce0(nn.Module):
    def __init__(self):
        super(Unet_avgpool_reduce0, self).__init__()
        self.conv1 = Conv(4, 64)
        self.down1 = DownSampling_avg()

        self.conv2 = Conv(64, 128)
        self.down2 = DownSampling_avg()

        self.conv3 = Conv(128, 256)
        self.down3 = DownSampling_avg()

        self.conv4 = Conv(256, 512)
        self.down4 = DownSampling_avg()

        self.conv5 = Conv(512, 1024)

        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)

        self.conv6 = Conv(1024, 512)
        self.conv7 = Conv(512, 256)
        self.conv8 = Conv(256, 128)
        self.conv9 = Conv(64, 32)  # changed!!!
        self.conv10 = nn.Conv2d(32, 1, 1, 1, 0)

    def forward(self, x):
        # level1
        x1 = self.conv1(x)
        x2 = self.down1(x1)

        # level2
        x2 = self.conv2(x2)
        x3 = self.down2(x2)

        # level3
        x3 = self.conv3(x3)
        x4 = self.down3(x3)

        # level4
        x4 = self.conv4(x4)
        x5 = self.down4(x4)

        # level5
        x5 = self.conv5(x5)
        y4 = self.up1(x5)

        # level4
        t4 = crop_tensor(x4, y4)
        y4 = torch.cat([t4, y4], dim=1)
        y4 = self.conv6(y4)
        y3 = self.up2(y4)

        # level3
        t3 = crop_tensor(x3, y3)
        y3 = torch.cat([t3, y3], dim=1)
        y3 = self.conv7(y3)
        y2 = self.up3(y3)

        # level2
        t2 = crop_tensor(x2, y2)
        y2 = torch.cat([t2, y2], dim=1)
        y2 = self.conv8(y2)
        y1 = self.up4(y2)

        # level1
        # t1 = crop_tensor(x1, y1)
        # y1 = torch.cat([t1, y1], dim=1)
        y1 = self.conv9(y1)

        U_out = self.conv10(y1)

        return U_out


class Unet_avgpool_reduce01(nn.Module):
    def __init__(self):
        super(Unet_avgpool_reduce01, self).__init__()
        self.conv1 = Conv(4, 64)
        self.down1 = DownSampling_avg()

        self.conv2 = Conv(64, 128)
        self.down2 = DownSampling_avg()

        self.conv3 = Conv(128, 256)
        self.down3 = DownSampling_avg()

        self.conv4 = Conv(256, 512)
        self.down4 = DownSampling_avg()

        self.conv5 = Conv(512, 1024)

        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(64, 32)

        self.conv6 = Conv(1024, 512)
        self.conv7 = Conv(512, 256)
        self.conv8 = Conv(128, 64)
        self.conv9 = Conv(32, 16)  # changed!!!
        self.conv10 = nn.Conv2d(16, 1, 1, 1, 0)

    def forward(self, x):
        # level1
        x1 = self.conv1(x)
        x2 = self.down1(x1)

        # level2
        x2 = self.conv2(x2)
        x3 = self.down2(x2)

        # level3
        x3 = self.conv3(x3)
        x4 = self.down3(x3)

        # level4
        x4 = self.conv4(x4)
        x5 = self.down4(x4)

        # level5
        x5 = self.conv5(x5)
        y4 = self.up1(x5)

        # level4
        t4 = crop_tensor(x4, y4)
        y4 = torch.cat([t4, y4], dim=1)
        y4 = self.conv6(y4)
        y3 = self.up2(y4)

        # level3
        t3 = crop_tensor(x3, y3)
        y3 = torch.cat([t3, y3], dim=1)
        y3 = self.conv7(y3)
        y2 = self.up3(y3)

        # level2
        # t2 = crop_tensor(x2, y2)
        # y2 = torch.cat([t2, y2], dim=1)
        y2 = self.conv8(y2)
        y1 = self.up4(y2)

        # level1
        # t1 = crop_tensor(x1, y1)
        # y1 = torch.cat([t1, y1], dim=1)
        y1 = self.conv9(y1)

        U_out = self.conv10(y1)

        return U_out


class Unet_avgpool_conv31(nn.Module):
    def __init__(self):
        super(Unet_avgpool_conv31, self).__init__()
        self.conv1 = Conv31(4, 64)
        self.down1 = DownSampling_avg()

        self.conv2 = Conv31(64, 128)
        self.down2 = DownSampling_avg()

        self.conv3 = Conv31(128, 256)
        self.down3 = DownSampling_avg()

        self.conv4 = Conv31(256, 512)
        self.down4 = DownSampling_avg()

        self.conv5 = Conv31(512, 1024)

        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)

        self.conv6 = Conv31(1024, 512)
        self.conv7 = Conv31(512, 256)
        self.conv8 = Conv31(256, 128)
        self.conv9 = Conv31(128, 64)
        self.conv10 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        # level1
        x1 = self.conv1(x)
        x2 = self.down1(x1)

        # level2
        x2 = self.conv2(x2)
        x3 = self.down2(x2)

        # level3
        x3 = self.conv3(x3)
        x4 = self.down3(x3)

        # level4
        x4 = self.conv4(x4)
        x5 = self.down4(x4)

        # level5
        x5 = self.conv5(x5)
        y4 = self.up1(x5)

        # level4
        t4 = crop_tensor(x4, y4)
        y4 = torch.cat([t4, y4], dim=1)
        y4 = self.conv6(y4)
        y3 = self.up2(y4)

        # level3
        t3 = crop_tensor(x3, y3)
        y3 = torch.cat([t3, y3], dim=1)
        y3 = self.conv7(y3)
        y2 = self.up3(y3)

        # level2
        t2 = crop_tensor(x2, y2)
        y2 = torch.cat([t2, y2], dim=1)
        y2 = self.conv8(y2)
        y1 = self.up4(y2)

        # level1
        t1 = crop_tensor(x1, y1)
        y1 = torch.cat([t1, y1], dim=1)
        y1 = self.conv9(y1)

        U_out = self.conv10(y1)

        return U_out


class Unet_avgpool_attention(nn.Module):
    def __init__(self):
        super(Unet_avgpool_attention, self).__init__()
        self.conv1 = Conv(4, 64)
        self.cbam1 = SpatialAttention(3)
        self.down1 = DownSampling_avg()

        self.conv2 = Conv(64, 128)
        self.cbam2 = SpatialAttention(3)
        self.down2 = DownSampling_avg()

        self.conv3 = Conv(128, 256)
        self.cbam3 = SpatialAttention(3)
        self.down3 = DownSampling_avg()

        self.conv4 = Conv(256, 512)
        self.cbam4 = SpatialAttention(3)
        self.down4 = DownSampling_avg()

        self.conv5 = Conv(512, 1024)
        self.cbam5 = SpatialAttention(3)

        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)

        self.conv6 = Conv(1024, 512)
        self.conv7 = Conv(512, 256)
        self.conv8 = Conv(256, 128)
        self.conv9 = Conv(128, 64)
        self.conv10 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        # level1
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x1 = self.cbam1(x1)

        # level2
        x2 = self.conv2(x2)
        x3 = self.down2(x2)
        x2 = self.cbam2(x2)

        # level3
        x3 = self.conv3(x3)
        x4 = self.down3(x3)
        x3 = self.cbam3(x3)

        # level4
        x4 = self.conv4(x4)
        x5 = self.down4(x4)
        x4 = self.cbam4(x4)

        # level5
        x5 = self.conv5(x5)
        x5 = self.cbam5(x5)
        y4 = self.up1(x5)

        # level4
        t4 = crop_tensor(x4, y4)
        y4 = torch.cat([t4, y4], dim=1)
        y4 = self.conv6(y4)
        y3 = self.up2(y4)

        # level3
        t3 = crop_tensor(x3, y3)
        y3 = torch.cat([t3, y3], dim=1)
        y3 = self.conv7(y3)
        y2 = self.up3(y3)

        # level2
        t2 = crop_tensor(x2, y2)
        y2 = torch.cat([t2, y2], dim=1)
        y2 = self.conv8(y2)
        y1 = self.up4(y2)

        # level1
        t1 = crop_tensor(x1, y1)
        y1 = torch.cat([t1, y1], dim=1)
        y1 = self.conv9(y1)

        U_out = self.conv10(y1)

        return U_out


class Unet_avgpool_level0123(nn.Module):
    def __init__(self):
        super(Unet_avgpool_level0123, self).__init__()
        self.conv1 = Conv(4, 64)
        self.down1 = DownSampling_avg()

        self.conv2 = Conv(64, 128)
        self.down2 = DownSampling_avg()

        self.conv3 = Conv(128, 256)
        self.down3 = DownSampling_avg()

        self.conv4 = Conv(256, 512)

        self.up1 = UpSampling(512, 256)
        self.up2 = UpSampling(256, 128)
        self.up3 = UpSampling(128, 64)

        self.conv5 = Conv(512, 256)
        self.conv6 = Conv(256, 128)
        self.conv7 = Conv(128, 64)
        self.conv8 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        # level1
        x1 = x
        x1 = self.conv1(x1)
        x2 = self.down1(x1)

        # level2
        x2 = self.conv2(x2)
        x3 = self.down2(x2)

        # level3
        x3 = self.conv3(x3)
        x4 = self.down3(x3)

        # # level4
        # x4 = self.conv4(x4)
        # x5 = self.down4(x4)

        # level5
        x4 = self.conv4(x4)
        y3 = self.up1(x4)

        # level4
        t3 = crop_tensor(x3, y3)
        y3 = torch.cat([t3, y3], dim=1)
        y3 = self.conv5(y3)
        y2 = self.up2(y3)

        # level3
        t2 = crop_tensor(x2, y2)
        y2 = torch.cat([t2, y2], dim=1)
        y2 = self.conv6(y2)
        y1 = self.up3(y2)

        # level2
        t1 = crop_tensor(x1, y1)
        y1 = torch.cat([t1, y1], dim=1)
        y1 = self.conv7(y1)
        y1 = self.up4(y1)

        # level1
        # t1 = crop_tensor(x1, y1)
        # y1 = torch.cat([t1, y1], dim=1)
        # y1 = self.conv9(y1)

        U_out = self.conv8(y1)

        return U_out


class Unet_avgpool_add(nn.Module):
    def __init__(self):
        super(Unet_avgpool_add, self).__init__()
        self.conv1 = Conv(4, 64)
        self.down1 = DownSampling_avg()

        self.conv2 = Conv(64, 128)
        self.down2 = DownSampling_avg()

        self.conv3 = Conv(128, 256)
        self.down3 = DownSampling_avg()

        self.conv4 = Conv(256, 512)
        self.down4 = DownSampling_avg()

        self.conv5 = Conv(512, 1024)

        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)

        self.conv6 = Conv(512, 512)
        self.conv7 = Conv(256, 256)
        self.conv8 = Conv(128, 128)
        self.conv9 = Conv(64, 64)
        self.conv10 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        # level1
        x1 = self.conv1(x)
        x2 = self.down1(x1)

        # level2
        x2 = self.conv2(x2)
        x3 = self.down2(x2)

        # level3
        x3 = self.conv3(x3)
        x4 = self.down3(x3)

        # level4
        x4 = self.conv4(x4)
        x5 = self.down4(x4)

        # level5
        x5 = self.conv5(x5)
        y4 = self.up1(x5)

        # level4
        # t4 = crop_tensor(x4, y4)
        # y4 = torch.cat([t4, y4], dim=1)
        y4 = y4 + x4
        y4 = self.conv6(y4)
        y3 = self.up2(y4)

        # level3
        # t3 = crop_tensor(x3, y3)
        # y3 = torch.cat([t3, y3], dim=1)
        y3 = y3 + x3
        y3 = self.conv7(y3)
        y2 = self.up3(y3)

        # level2
        # t2 = crop_tensor(x2, y2)
        # y2 = torch.cat([t2, y2], dim=1)
        y2 = y2 + x2
        y2 = self.conv8(y2)
        y1 = self.up4(y2)

        # level1
        # t1 = crop_tensor(x1, y1)
        # y1 = torch.cat([t1, y1], dim=1)
        y1 = y1 + x1
        y1 = self.conv9(y1)

        U_out = self.conv10(y1)

        return U_out


class Unet_avgpool_add_level012345(nn.Module):
    def __init__(self):
        super(Unet_avgpool_add_level012345, self).__init__()
        self.conv1 = Conv(4, 64)
        self.down1 = DownSampling_avg()

        self.conv2 = Conv(64, 128)
        self.down2 = DownSampling_avg()

        self.conv3 = Conv(128, 256)
        self.down3 = DownSampling_avg()

        self.conv4 = Conv(256, 512)
        self.down4 = DownSampling_avg()

        self.conv5 = Conv(512, 1024)
        self.down5 = DownSampling_avg()

        self.conv6_0 = Conv(1024, 2048)

        self.up0_0 = UpSampling(2048, 1024)
        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)

        self.conv6_1 = Conv(1024, 1024)
        self.conv6 = Conv(512, 512)
        self.conv7 = Conv(256, 256)
        self.conv8 = Conv(128, 128)
        self.conv9 = Conv(64, 64)
        self.conv10 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        # level1
        x1 = self.conv1(x)
        x2 = self.down1(x1)

        # level2
        x2 = self.conv2(x2)
        x3 = self.down2(x2)

        # level3
        x3 = self.conv3(x3)
        x4 = self.down3(x3)

        # level4
        x4 = self.conv4(x4)
        x5 = self.down4(x4)

        # level5
        x5 = self.conv5(x5)
        x6 = self.down5(x5)

        # level6
        x6 = self.conv6_0(x6)
        y5 = self.up0_0(x6)

        # level5
        y5 = y5 + x5
        y5 = self.conv6_1(y5)
        y4 = self.up1(y5)

        # level4
        # t4 = crop_tensor(x4, y4)
        # y4 = torch.cat([t4, y4], dim=1)
        y4 = y4 + x4
        y4 = self.conv6(y4)
        y3 = self.up2(y4)

        # level3
        # t3 = crop_tensor(x3, y3)
        # y3 = torch.cat([t3, y3], dim=1)
        y3 = y3 + x3
        y3 = self.conv7(y3)
        y2 = self.up3(y3)

        # level2
        # t2 = crop_tensor(x2, y2)
        # y2 = torch.cat([t2, y2], dim=1)
        y2 = y2 + x2
        y2 = self.conv8(y2)
        y1 = self.up4(y2)

        # level1
        # t1 = crop_tensor(x1, y1)
        # y1 = torch.cat([t1, y1], dim=1)
        y1 = y1 + x1
        y1 = self.conv9(y1)

        U_out = self.conv10(y1)

        return U_out


class Unet_avgpool_add_level012345_depth(nn.Module):
    def __init__(self):
        super(Unet_avgpool_add_level012345_depth, self).__init__()
        self.conv1 = Conv(5, 64)
        self.down1 = DownSampling_avg()

        self.conv2 = Conv(64, 128)
        self.down2 = DownSampling_avg()

        self.conv3 = Conv(128, 256)
        self.down3 = DownSampling_avg()

        self.conv4 = Conv(256, 512)
        self.down4 = DownSampling_avg()

        self.conv5 = Conv(512, 1024)
        self.down5 = DownSampling_avg()

        self.conv6_0 = Conv(1024, 2048)

        self.up0_0 = UpSampling(2048, 1024)
        self.up1 = UpSampling(1024, 512)
        self.up2 = UpSampling(512, 256)
        self.up3 = UpSampling(256, 128)
        self.up4 = UpSampling(128, 64)

        self.conv6_1 = Conv(1024, 1024)
        self.conv6 = Conv(512, 512)
        self.conv7 = Conv(256, 256)
        self.conv8 = Conv(128, 128)
        self.conv9 = Conv(64, 64)
        self.conv10 = nn.Conv2d(64, 1, 1, 1, 0)

    def forward(self, x):
        # level1
        x1 = self.conv1(x)
        x2 = self.down1(x1)

        # level2
        x2 = self.conv2(x2)
        x3 = self.down2(x2)

        # level3
        x3 = self.conv3(x3)
        x4 = self.down3(x3)

        # level4
        x4 = self.conv4(x4)
        x5 = self.down4(x4)

        # level5
        x5 = self.conv5(x5)
        x6 = self.down5(x5)

        # level6
        x6 = self.conv6_0(x6)
        y5 = self.up0_0(x6)

        # level5
        y5 = y5 + x5
        y5 = self.conv6_1(y5)
        y4 = self.up1(y5)

        # level4
        # t4 = crop_tensor(x4, y4)
        # y4 = torch.cat([t4, y4], dim=1)
        y4 = y4 + x4
        y4 = self.conv6(y4)
        y3 = self.up2(y4)

        # level3
        # t3 = crop_tensor(x3, y3)
        # y3 = torch.cat([t3, y3], dim=1)
        y3 = y3 + x3
        y3 = self.conv7(y3)
        y2 = self.up3(y3)

        # level2
        # t2 = crop_tensor(x2, y2)
        # y2 = torch.cat([t2, y2], dim=1)
        y2 = y2 + x2
        y2 = self.conv8(y2)
        y1 = self.up4(y2)

        # level1
        # t1 = crop_tensor(x1, y1)
        # y1 = torch.cat([t1, y1], dim=1)
        y1 = y1 + x1
        y1 = self.conv9(y1)

        U_out = self.conv10(y1)

        return U_out


class UnetPP(nn.Module):
    def __init__(self):
        super(UnetPP, self).__init__()

# if __name__ == '__main__':
#     unet = Unet_Bilinear()
#     input = torch.rand(1, 1, 1024, 2048)
#     output = unet(input)
#     print(output.shape)
