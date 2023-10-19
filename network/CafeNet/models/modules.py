import torch
from torch import nn
from torch.nn import functional as F
from network.CafeNet.models.resnet import resnet50

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=False, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_c, out_c,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias
            ),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x


""" Squeeze and Excitation block """
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = Conv2D(in_c, out_c)
        self.conv2 = Conv2D(out_c, out_c, act=False)
        self.conv3 = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)
        self.se = SELayer(out_c, out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x2 = self.se(x2)

        x3 = self.conv3(x)

        x4 = x2 + x3
        x4 = self.relu(x4)
        return x4


class GCBlock(nn.Module):
    """
      Paper title: <GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond>
      Paper link: https://arxiv.org/abs/1904.11492
      Paper accepted by CVPR 2019
    """
    def __init__(self, inplanes, planes, pool, fusions):
        super(GCBlock, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = inplanes
        self.planes = planes
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()  # N C H W
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask) # [N, 1, C, H * W]  [N, 1, H * W, 1] -> [N, 1, C, 1]
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class MCA(nn.Module):
    def __init__(self, in_c, out_c):
        super(MCA, self).__init__()

        self.conv1 = Conv2D(in_c, in_c//4, kernel_size=3, padding=1, dilation=1)
        self.conv2 = Conv2D(in_c, in_c//4, kernel_size=3, padding=3, dilation=3)
        self.conv3 = Conv2D(in_c, in_c//4, kernel_size=3, padding=5, dilation=5)
        self.conv4 = Conv2D(in_c, in_c//4, kernel_size=1, padding=0)
        self.gc = GCBlock(in_c//4, in_c//4, 'att', ['channel_add'])
        self.seBlock = SEBlock((in_c//4)*4, out_c)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x4 = self.conv4(x)
        x4 = self.gc(x4)

        x5 = torch.cat([x1,x2,x3,x4], dim=1)

        output = self.seBlock(x5)
        return output


class MFE(nn.Module):
    def __init__(self, in_c):
        super(MFE, self).__init__()
        self.convBlock = nn.Sequential(nn.Conv2d(in_c, in_c, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(in_c),
                                   nn.ReLU(inplace=True))


    def forward(self, x, y, predict):
        x1 = self.convBlock(x)
        x2 = self.convBlock(y)
        x3 = self.convBlock(abs(F.interpolate(x2, size=x1.size()[2:], mode='bilinear') - x1))

        p = F.interpolate(predict, size=x3.size()[2:], mode='bilinear')
        p = torch.sigmoid(p)
        output = p.expand(-1, x3.size()[1], -1, -1).mul(x3)
        output = self.convBlock(output)
        return output


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

