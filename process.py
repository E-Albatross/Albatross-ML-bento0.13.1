from collections import OrderedDict
# from craft_model import CRAFT
import segmentation_models_pytorch as smp

import torch

import torch.nn as nn
import torch.nn.functional as F

from vgg16_bn import vgg16_bn, init_weights

class CraftMain():
    def __init__(self):
        self.model = CRAFT(pretrained=True) # initialize

    def load_model(self, checkpoint, cuda=False):
        if cuda:
            print('craft model loaded with cuda')
            self.model.load_state_dict(self.copyStateDict(torch.load(checkpoint)))
        else:
            self.model.load_state_dict(self.copyStateDict(torch.load(checkpoint, map_location='cpu')))

        return self.model

    def copyStateDict(self,state_dict):
        if list(state_dict.keys())[0].startswith("module"):
            start_idx = 1
        else:
            start_idx = 0
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = ".".join(k.split(".")[start_idx:])
            new_state_dict[name] = v
        return new_state_dict



# class FPNMain():
#     def __init__(self, backbone='resnext50', n_class=3):
#         self.backbone = backbone
#         self.n_class = n_class
#         self.model =  FPN(encoder_name=self.backbone,
#                 decoder_pyramid_channels=256,
#                 decoder_segmentation_channels=128,
#                 classes=self.n_class,
#                 dropout=0.3,
#                 activation='sigmoid',
#                 final_upsampling=4,
#                 decoder_merge_policy='add')## Optimizer 설정
#
#     def load_model(self, checkpoint, cuda=False):
#         if cuda:
#             state = torch.load(checkpoint)
#         else:
#             state = torch.load(checkpoint, map_location=torch.device('cpu'))
#         self.model.load_state_dict(state['state_dict'])
#
#         return self.model

class SegmentationMain():
    def __init__(self):
        self.model = smp.FPN(encoder_name="resnext50_32x4d", classes=3)

    def load_model(self, checkpoint, cuda=False):
        if cuda:
            print('segmentation model loaded with cuda')
            state = torch.load(checkpoint)
        else:
            state = torch.load(checkpoint, map_location=torch.device('cpu'))
        self.model.load_state_dict(state['model_state_dict'])

        return self.model


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class CRAFT(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(CRAFT, self).__init__()

        """ Base network """
        self.basenet = vgg16_bn(pretrained, freeze)

        """ U network """
        self.upconv1 = double_conv(1024, 512, 256)
        self.upconv2 = double_conv(512, 256, 128)
        self.upconv3 = double_conv(256, 128, 64)
        self.upconv4 = double_conv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature