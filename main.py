import argparse

# from process import *
from service import HangeulDetector
from service import CraftMain
from service import SegmentationMain

from collections import OrderedDict
# from craft_model import CRAFT
import segmentation_models_pytorch as smp

import torch

import torch.nn as nn
import torch.nn.functional as F

from collections import namedtuple

import torch.nn.init as init
from torchvision import models
from torchvision.models.vgg import model_urls

# def init_weights(modules):
#     for m in modules:
#         if isinstance(m, nn.Conv2d):
#             init.xavier_uniform_(m.weight.data)
#             if m.bias is not None:
#                 m.bias.data.zero_()
#         elif isinstance(m, nn.BatchNorm2d):
#             m.weight.data.fill_(1)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.Linear):
#             m.weight.data.normal_(0, 0.01)
#             m.bias.data.zero_()
#
# class vgg16_bn(torch.nn.Module):
#     def __init__(self, pretrained=True, freeze=True):
#         super(vgg16_bn, self).__init__()
#         model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
#         vgg_pretrained_features = models.vgg16_bn(pretrained=pretrained).features
#         self.slice1 = torch.nn.Sequential()
#         self.slice2 = torch.nn.Sequential()
#         self.slice3 = torch.nn.Sequential()
#         self.slice4 = torch.nn.Sequential()
#         self.slice5 = torch.nn.Sequential()
#         for x in range(12):         # conv2_2
#             self.slice1.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(12, 19):         # conv3_3
#             self.slice2.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(19, 29):         # conv4_3
#             self.slice3.add_module(str(x), vgg_pretrained_features[x])
#         for x in range(29, 39):         # conv5_3
#             self.slice4.add_module(str(x), vgg_pretrained_features[x])
#
#         # fc6, fc7 without atrous conv
#         self.slice5 = torch.nn.Sequential(
#                 nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
#                 nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
#                 nn.Conv2d(1024, 1024, kernel_size=1)
#         )
#
#         if not pretrained:
#             init_weights(self.slice1.modules())
#             init_weights(self.slice2.modules())
#             init_weights(self.slice3.modules())
#             init_weights(self.slice4.modules())
#
#         init_weights(self.slice5.modules())        # no pretrained model for fc6 and fc7
#
#         if freeze:
#             for param in self.slice1.parameters():      # only first conv
#                 param.requires_grad= False
#
#     def forward(self, X):
#         h = self.slice1(X)
#         h_relu2_2 = h
#         h = self.slice2(h)
#         h_relu3_2 = h
#         h = self.slice3(h)
#         h_relu4_3 = h
#         h = self.slice4(h)
#         h_relu5_3 = h
#         h = self.slice5(h)
#         h_fc7 = h
#         vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
#         out = vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)
#         return out
#
# class CraftMain():
#     def __init__(self):
#         self.model = CRAFT() # initialize
#
#     def load_model(self, checkpoint, cuda=False):
#         if cuda:
#             self.model.load_state_dict(self.copyStateDict(torch.load(checkpoint)))
#         else:
#             self.model.load_state_dict(self.copyStateDict(torch.load(checkpoint, map_location='cpu')))
#
#         return self.model
#
#     def copyStateDict(self,state_dict):
#         if list(state_dict.keys())[0].startswith("module"):
#             start_idx = 1
#         else:
#             start_idx = 0
#         new_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             name = ".".join(k.split(".")[start_idx:])
#             new_state_dict[name] = v
#         return new_state_dict
#
#
#
# # class FPNMain():
# #     def __init__(self, backbone='resnext50', n_class=3):
# #         self.backbone = backbone
# #         self.n_class = n_class
# #         self.model =  FPN(encoder_name=self.backbone,
# #                 decoder_pyramid_channels=256,
# #                 decoder_segmentation_channels=128,
# #                 classes=self.n_class,
# #                 dropout=0.3,
# #                 activation='sigmoid',
# #                 final_upsampling=4,
# #                 decoder_merge_policy='add')## Optimizer 설정
# #
# #     def load_model(self, checkpoint, cuda=False):
# #         if cuda:
# #             state = torch.load(checkpoint)
# #         else:
# #             state = torch.load(checkpoint, map_location=torch.device('cpu'))
# #         self.model.load_state_dict(state['state_dict'])
# #
# #         return self.model
#
# class SegmentationMain():
#     def __init__(self):
#         self.model = smp.FPN(encoder_name="resnext50_32x4d", classes=3)
#
#     def load_model(self, checkpoint, cuda=False):
#         if cuda:
#             state = torch.load(checkpoint)
#         else:
#             state = torch.load(checkpoint, map_location=torch.device('cpu'))
#         self.model.load_state_dict(state['model_state_dict'])
#
#         return self.model
#
#
# class double_conv(nn.Module):
#     def __init__(self, in_ch, mid_ch, out_ch):
#         super(double_conv, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
#             nn.BatchNorm2d(mid_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x
#
#
# class CRAFT(nn.Module):
#     def __init__(self, pretrained=False, freeze=False):
#         super(CRAFT, self).__init__()
#
#         """ Base network """
#         self.basenet = vgg16_bn(pretrained, freeze)
#
#         """ U network """
#         self.upconv1 = double_conv(1024, 512, 256)
#         self.upconv2 = double_conv(512, 256, 128)
#         self.upconv3 = double_conv(256, 128, 64)
#         self.upconv4 = double_conv(128, 64, 32)
#
#         num_class = 2
#         self.conv_cls = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
#             nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
#             nn.Conv2d(16, num_class, kernel_size=1),
#         )
#
#         init_weights(self.upconv1.modules())
#         init_weights(self.upconv2.modules())
#         init_weights(self.upconv3.modules())
#         init_weights(self.upconv4.modules())
#         init_weights(self.conv_cls.modules())
#
#     def forward(self, x):
#         """ Base network """
#         sources = self.basenet(x)
#
#         """ U network """
#         y = torch.cat([sources[0], sources[1]], dim=1)
#         y = self.upconv1(y)
#
#         y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
#         y = torch.cat([y, sources[2]], dim=1)
#         y = self.upconv2(y)
#
#         y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
#         y = torch.cat([y, sources[3]], dim=1)
#         y = self.upconv3(y)
#
#         y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
#         y = torch.cat([y, sources[4]], dim=1)
#         feature = self.upconv4(y)
#
#         y = self.conv_cls(feature)
#
#         return y.permute(0, 2, 3, 1), feature

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, help='enable CUDA')
    parser.add_argument("--craft_model", default="./craft_mlt_25k.pth", help='text detection model path')
    parser.add_argument("--fpn_model", default="./fpn_last.pth", help='segmentation model path')
    parser.add_argument("--seg_model", default="./seg_model.pth", help='segmentation model path')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    craft = CraftMain()
    seg = SegmentationMain()
    # fpn = FPNMain()

    # load model
    craft_model = craft.load_model(args.craft_model, device)
    segmentation_model = seg.load_model(args.seg_model, device)
    # fpn_model = fpn.load_model(args.fpn_model, cuda)

    detector_service = HangeulDetector()
    detector_service.pack('craft_model', craft_model)
    detector_service.pack('fpn_model', segmentation_model)


    saved_path = detector_service.save()

