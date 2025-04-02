import cv2

from NetStructure import *
from MyDataset import *
from Loss import *

import torch
from torch.utils.data import DataLoader
import torch.optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
import os
import json

class NSM(nn.Module):
    def __init__(self):
        super(NSM, self).__init__()
        self.conv0_0 = Conv31(4,16)
        self.conv1_0 = Conv31(16,16)
        self.conv2_0 = Conv31(16,32)
        self.conv3_0 = Conv31(32,64)

        self.conv4_0 = nn.Conv2d(64,256,3,1,1)
        self.conv4_1 = nn.Conv2d(256,64,3,1,1)

        self.down0 = DownSampling_avg()
        self.down1 = DownSampling_avg()
        self.down2 = DownSampling_avg()
        self.down3 = DownSampling_avg()

        self.conv0_1 = Conv31(16,1)
        self.conv1_1 = Conv31(16,16)
        self.conv2_1 = Conv31(32,16)
        self.conv3_1 = Conv31(64,32)

        self.up0 = UpBilinear()
        self.up1 = UpBilinear()
        self.up2 = UpBilinear()
        self.up3 = UpBilinear()



    def forward(self, x):
        x0 = x
        # down
        x0 = self.conv0_0(x0)
        x1 = self.down0(x0)

        x1 = self.conv1_0(x1)
        x2 = self.down1(x1)

        x2 = self.conv2_0(x2)
        x3 = self.down2(x2)

        x3 = self.conv3_0(x3)
        x4 = self.down3(x3)

        x4 = self.conv4_0(x4)
        y4 = self.conv4_1(x4)

        # up
        y3 = self.up0(y4)
        y3 = y3 + x3
        y3 = self.conv3_1(y3)

        y2 = self.up1(y3)
        y2 = y2 + x2
        y2 = self.conv2_1(y2)

        y1 = self.up2(y2)
        y1 = y1 + x1
        y1 = self.conv1_1(y3)

        y0 = self.up3(y1)
        # y0 = y0 + x0
        y0 = self.conv0_1(y3)

        return y0

def train_net(net, device, config, epochs, batch_size, lr, last_epoch=0):


if __name__ == '__main__':
    srcImg = torch.tensor([[[[1, 2, 3],
                                   [4, 5, 6]]]], dtype=torch.float32)
    up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
    dstImg = up(srcImg)
    print(srcImg)
    print(dstImg)

