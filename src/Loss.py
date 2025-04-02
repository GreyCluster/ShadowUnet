from torch import nn
import torch
from torchvision import models


class VGG(nn.Module):
    def __init__(self, device, n_layers=5):
        super(VGG, self).__init__()
        self.device = device
        self.alpha = 0.9
        self.l1_loss = nn.L1Loss().to(device)

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = models.vgg19(pretrained=True).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, input, gt):
        input = input.to(self.device)
        gt = gt.to(self.device)

        # l1 loss must be earlier than vgg!!!
        l1_loss = self.l1_loss(gt, input)

        # vgg loss
        if input.size(1) == 1:
            input = input.repeat(1, 3, 1, 1)
        if gt.size(1) == 1:
            gt = gt.repeat(1, 3, 1, 1)

        vgg_loss = 0
        for layer, weight in zip(self.layers, self.weights):
            input = layer(input)
            with torch.no_grad():
                gt = layer(gt)
            vgg_loss += weight * self.l1_loss(input, gt)

        loss = self.alpha * l1_loss + (1 - self.alpha) * vgg_loss

        return loss

# if __name__ == '__main__':
#     input = torch.rand(1,1,3,4)
#     gt = torch.rand(1,1,3,4)
#
#     ls_fn = nn.L1Loss()
#     loss1 = ls_fn(input,gt)
#
#     input = input.repeat(1,3,1,1)
#     gt = gt.repeat(1,3,1,1)
#     loss3 = ls_fn(input, gt)
#
#     print("loss1:{},loss3{}".format(loss1, loss3))
