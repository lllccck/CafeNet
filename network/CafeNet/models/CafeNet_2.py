import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from matplotlib import pyplot as plt

from network.CafeNet.models.resnet import resnet50
from network.CafeNet.models.modules import Conv2D, SEBlock, MCA, MFE, ChannelAttention, SpatialAttention
from config import CONFIG
import seaborn as sns

class lowDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(lowDecoderBlock, self).__init__()
        self.seBlock = SEBlock(in_c, out_c)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.seBlock(x)
        x = self.upsample(x)
        return x


class highDecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(highDecoderBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seBlock = SEBlock(in_c[0]+in_c[1], out_c)

    def forward(self, x, s):
        x = torch.cat([x, s], axis=1)
        x = self.seBlock(x)
        x = self.upsample(x)
        return x


class predictBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1):
        super(predictBlock, self).__init__()

        self.conv1 = Conv2D(in_c, in_c // 4, kernel_size=kernel_size, padding=padding, act=True)

        self.dropout = nn.Dropout2d(0.1)

        self.conv2 = nn.Conv2d(in_c // 4, out_c, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class CafeNet(nn.Module):
    def __init__(self, num_classes):
        super(CafeNet, self).__init__()

        """ Backbone: ResNet50 """
        backbone = resnet50()
        self.encoder1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.encoder2 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.encoder3 = backbone.layer2
        self.encoder4 = backbone.layer3

        self.seBlock1 = SEBlock(64, 96)
        self.seBlock2 = SEBlock(256, 96)
        self.seBlock3 = SEBlock(512, 96)
        self.seBlock4 = SEBlock(1024, 96)

        self.decoder4 = lowDecoderBlock(96, 96)
        self.decoder3 = highDecoderBlock([96, 96], 96)
        self.decoder2 = highDecoderBlock([96, 96], 96)
        self.decoder1 = highDecoderBlock([96, 96], 96)

        self.predict1 = predictBlock(96, 1)
        self.predict2 = predictBlock(96, 1)
        self.predict3 = predictBlock(96, 1)

        self.mca1 = MCA(96, 96)
        self.mca2 = MCA(96, 96)
        self.mca3 = MCA(96, 96)
        self.mca4 = MCA(96, 96)

        self.mfe1 = MFE(96)
        self.mfe2 = MFE(96)
        self.mfe3 = MFE(96)

        self.ca = ChannelAttention(96)
        self.sa = SpatialAttention()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.out = nn.Conv2d(96, num_classes, kernel_size=1, padding=0)

    def forward(self, image):
        e1 = self.encoder1(image)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        se1 = self.seBlock1(e1)
        se2 = self.seBlock2(e2)
        se3 = self.seBlock3(e3)
        se4 = self.seBlock4(e4)

        mca1 = self.mca1(se1)
        mca2 = self.mca2(se2)
        mca3 = self.mca3(se3)
        mca4 = self.mca4(se4)

        d4 = self.decoder4(mca4)
        p1 = self.predict1(d4)
        mfe1 = self.mfe1(mca3, mca4, p1)
        add_out1 = mca3 + mfe1
        mfe1 = self.upsample(mfe1)

        d3 = self.decoder3(d4, add_out1)
        p2 = self.predict2(d3)
        mfe2 = self.mfe2(mca2, mca3, p2)
        add_out2 = mca2 + mfe2 + mfe1
        mfe2 = self.upsample(mfe2)

        d2 = self.decoder2(d3, add_out2)
        p3 = self.predict3(d2)
        mfe3 = self.mfe3(mca1, mca2, p3)
        mfe1 = self.upsample(mfe1)
        add_out3 = mca1 + mfe3 + mfe2 + mfe1

        d1 = self.decoder1(d2, add_out3)

        d1 = d1 * self.ca(d1)
        d1 = d1 * self.sa(d1)

        out = self.out(d1)

        return torch.sigmoid(out), torch.sigmoid(p3), torch.sigmoid(p2), torch.sigmoid(p1)

class CafeNetModel(nn.Module):
  def __init__(self, config):
    super(CafeNetModel,self).__init__()
    self.config=config
    self.num_classes=self.config["models"]["num_classes"]
    self.net = CafeNet(self.num_classes)

  def forward(self, images):
    out1,out2,out3,out4= self.net(images)
    return out1,out2,out3,out4


if __name__ == "__main__":
    inputs = torch.randn((2, 3, 256, 256))
    model = CafeNetModel(CONFIG)
    out = model(inputs)
    print(out[0].shape)
