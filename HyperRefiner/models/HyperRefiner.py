# -*- coding：UTF-8 -*-
'''
@Project : HyperRefiner
@Author : Zerbo
@Date : 2022/9/16 15:39
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import cv2
import math
from models.AE import ConvAutoencoder as AE
from models.models import MODELS

BatchNorm2d = torch.nn.BatchNorm2d


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory)


RELUSLOPE = 0.2


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, res_scale=1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        x1 = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out * self.res_scale + x1
        return out


class SpectralAttention(nn.Module):
    ''' Spectral transformer '''

    def __init__(self, n_feats):
        super().__init__()
        self.num_features = n_feats

        self.query = nn.Sequential(nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=1, stride=1, padding=0, bias=False),
                                   BatchNorm2d(self.num_features), nn.ReLU())
        self.key = nn.Sequential(nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=1, stride=1, padding=0, bias=False),
                                 BatchNorm2d(self.num_features), nn.ReLU())
        self.value = nn.Sequential(nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=1, stride=1, padding=0, bias=False),
                                   BatchNorm2d(self.num_features), nn.ReLU())

        self.tail = nn.Sequential(nn.Conv2d(in_channels=self.num_features, out_channels=self.num_features, kernel_size=1, stride=1, padding=0, bias=False),
                                  BatchNorm2d(self.num_features), nn.ReLU())

    def forward(self, lr_hsi, refine_hsi):
        b, c, H, W = refine_hsi.size(0), refine_hsi.size(1), refine_hsi.size(2), refine_hsi.size(3)

        q = self.query(refine_hsi).view(b, self.num_features, -1).permute(0, 2, 1)  # b c HW  ----b HW c
        k = self.key(lr_hsi).view(b, self.num_features, -1)  # b c HW/4
        v = self.value(lr_hsi).view(b, self.num_features, -1).permute(0, 2, 1)  # b HW/4 c

        correlation = torch.matmul(q, k)  # b HW HW/4
        correlation = (self.num_features ** -0.5) * correlation
        correlation = F.softmax(correlation, dim=1)

        spatial_spectral = torch.matmul(correlation, v)  # b HW c
        spatial_spectral = spatial_spectral.permute(0, 2, 1).contiguous()  # b c HW
        spatial_spectral = spatial_spectral.view(b, self.num_features, H, W)
        spatial_spectral = self.tail(spatial_spectral)  # b c H W

        return spatial_spectral, correlation


class LFE(nn.Module):
    def __init__(self, in_channels):
        super(LFE, self).__init__()

        # Define number of input channels
        self.in_channels = in_channels
        self.filters = 64

        # First level convolutions
        self.conv_64_1 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.filters, kernel_size=3, padding=1)
        self.bn_64_1 = nn.BatchNorm2d(self.filters)
        self.conv_64_2 = nn.Conv2d(in_channels=self.filters, out_channels=self.filters, kernel_size=3, padding=1)
        self.bn_64_2 = nn.BatchNorm2d(self.filters)

        # Second level convolutions
        self.conv_128_1 = nn.Conv2d(in_channels=self.filters, out_channels=2 * self.filters, kernel_size=3, padding=1)
        self.bn_128_1 = nn.BatchNorm2d(2 * self.filters)
        self.conv_128_2 = nn.Conv2d(in_channels=2 * self.filters, out_channels=2 * self.filters, kernel_size=3, padding=1)
        self.bn_128_2 = nn.BatchNorm2d(2 * self.filters)

        # Third level convolutions
        self.conv_256_1 = nn.Conv2d(in_channels=2 * self.filters, out_channels=4 * self.filters, kernel_size=3, padding=1)
        self.bn_256_1 = nn.BatchNorm2d(4 * self.filters)
        self.conv_256_2 = nn.Conv2d(in_channels=4 * self.filters, out_channels=4 * self.filters, kernel_size=3, padding=1)
        self.bn_256_2 = nn.BatchNorm2d(4 * self.filters)

        # Max pooling
        self.MaxPool2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # LeakyReLU
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.0)

    def forward(self, x):
        # First level outputs
        # X：PAN B*1*H*W
        out1 = self.LeakyReLU(self.bn_64_1(self.conv_64_1(x)))  # B*128*H*W
        out1 = self.bn_64_2(self.conv_64_2(out1))  # B*128*H*W

        # Second level outputs
        out1_mp = self.MaxPool2x2(self.LeakyReLU(out1))
        out2 = self.LeakyReLU(self.bn_128_1(self.conv_128_1(out1_mp)))
        out2 = self.bn_128_2(self.conv_128_2(out2))  # B*256*0.5H*0.5W

        # Third level outputs
        out2_mp = self.MaxPool2x2(self.LeakyReLU(out2))
        out3 = self.LeakyReLU(self.bn_256_1(self.conv_256_1(out2_mp)))
        out3 = self.bn_256_2(self.conv_256_2(out3))  # B*512*0.25H*0.25W

        return out1, out2, out3


class HyperAE(nn.Module):
    "嵌入AE和sr模块，使用self-attention优化光谱"

    def __init__(self, config):
        super(HyperAE, self).__init__()

        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.HR_size = config[config["train_dataset"]]["HR_size"]
        self.LR_size = config[config["train_dataset"]]["LR_size"]
        self.LFE = LFE(self.in_channels + 1)
        self.channels = config["AE"]["channels"]
        self.upscale_method = config["upscale_method"]
        if self.upscale_method == "sr":
            self.sr = MODELS[config["sr"]["sr_model"]](config)
        self.AE = AE(config)
        self.factor = config[config["train_dataset"]]["factor"]
        self.flag=config["train"]

        self.num_res_blocks = [0, 0, 0, 0]
        #self.outchannels = [128, 256, 512]
        self.outchannels = [64, 128, 256]
        self.res_scale = 1

        # LR_size at different level from AE
        self.lv1_size = int(self.LR_size / 2)
        self.lv2_size = int(self.LR_size / 4)
        self.lv3_size = int(self.LR_size / 8)

        #
        self.spectralattention1 = SpectralAttention(n_feats=self.channels[0])  # c=128
        self.spectralattention2 = SpectralAttention(n_feats=self.channels[1])  # c=256
        self.spectralattention3 = SpectralAttention(n_feats=self.channels[2])  # c=512

        self.RB1 = nn.ModuleList()
        for i in range(self.num_res_blocks[0]):
            self.RB1.append(ResBlock(in_channels=self.outchannels[0], out_channels=self.outchannels[0], res_scale=self.res_scale))

        self.RB2 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB2.append(ResBlock(in_channels=self.outchannels[1], out_channels=self.outchannels[1], res_scale=self.res_scale))

        self.RB3 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB3.append(ResBlock(in_channels=self.outchannels[2], out_channels=self.outchannels[2], res_scale=self.res_scale))

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks[-1]):
            self.RBs.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))

        self.conv_tail1 = nn.Conv2d(in_channels=self.outchannels[0], out_channels=self.outchannels[0], kernel_size=3, padding=1)
        self.conv_tail2 = nn.Conv2d(in_channels=self.outchannels[1], out_channels=self.outchannels[1], kernel_size=3, padding=1)
        self.conv_tail3 = nn.Conv2d(in_channels=self.outchannels[2], out_channels=self.outchannels[2], kernel_size=3, padding=1)
        self.final = nn.Conv2d(in_channels=sum(self.outchannels), out_channels=self.out_channels, kernel_size=1, padding=0)
        self.conv_tailf = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1)

        self.conv32 = nn.Conv2d(in_channels=self.outchannels[2], out_channels=2 * self.outchannels[2], kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(in_channels=self.outchannels[1], out_channels=2 * self.outchannels[1], kernel_size=3, padding=1)

        self.ps32 = nn.PixelShuffle(2)
        self.ps21 = nn.PixelShuffle(2)

    def forward(self, PAN, X_MS, MS_dhp=None):
    
        if self.upscale_method == "interpolate" and MS_dhp is None:
            coarse_hsi = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        if self.upscale_method == "DHP" and MS_dhp is not None:
            coarse_hsi = MS_dhp
        if self.upscale_method == "sr" and MS_dhp is None:
            coarse_hsi = self.sr(PAN, X_MS)  # b, c, H, W



            PAN = PAN.unsqueeze(dim=1)
            b, c, h, w = X_MS.size()

        # 对X_MS进行自编码监督，压缩空间特征，提取光谱特征
        if self.flag:
            out, de_lr_hsi = self.AE(X_MS)  # en: b, c, h, w
        else:
            out = self.AE(X_MS)
            # 获取AE编码器前三层输出并reshape，每层输出空间尺度为H/2, H/4,H/8, 通道数分别为128，256，512
        en = []
        b, c1, hw = out[0].size()
        for i in range(3):
            entmp = out[i]  # b, c, hw
            h = int(h / 2)
            w = int(w / 2)
            en.append(entmp.view(b, c1, h, w))
            c1 = int(2 * c1)

        feats = torch.cat((coarse_hsi, PAN), dim=1)
        f1, f2, f3 = self.LFE(feats)  # 特征提取，三个空间尺度分别为H, H/2, H/4, 通道数分别为128，256，512
        #f111=f1
        t1, am1 = self.spectralattention1(en[0], f1)  # b, c, H, W
        t2, am2 = self.spectralattention2(en[1], f2)  # b, 2c, H/2, W/2
        t3, am3 = self.spectralattention3(en[2], f3)  # b, 4c, H/4, W/4

        # lv3, lr(L/8, W/8), hr(L/4, W/4) , t3--> b, 4c, H/4, W/4
        f3 = f3 + t3
        f3_res = f3
        for i in range(self.num_res_blocks[2]):
            f3 = self.RB3[i](f3)
        f3 = self.conv_tail3(f3)
        f3 = f3 + f3_res  # b, 4c, H/4, W/4

        # lv2, lr(L/4, W/4), hr(L/2, W/2) , t2--> b, 2c, H/2, W/2
        f22 = F.relu(self.ps32(self.conv32(f3)))  # b 2c H/2 W/2
        f2 = f2 + t2 + f22
        f2_res = f2
        for i in range(self.num_res_blocks[1]):
            f2 = self.RB2[i](f2)
        f2 = self.conv_tail2(f2)
        f2 = f2 + f2_res  # b 2c H/2 W/2

        # lv1, lr(L/2, W/2), hr(L, W) , t1--> b, c, H, W
        f11 = F.relu(self.ps21(self.conv21(f2)))  # b c H W
        f1 = f1 + t1 + f11
        f1_res = f1
        for i in range(self.num_res_blocks[0]):
            f1 = self.RB1[i](f1)
        f1 = self.conv_tail1(f1)
        f11 = f1 + f1_res

        f22 = F.interpolate(f2, scale_factor=2, mode='bicubic')
        f33 = F.interpolate(f3, scale_factor=4, mode='bicubic')

        f = torch.cat((f11, f22, f33), dim=1)
        f = self.final(f)
        ref = f
        for i in range(self.num_res_blocks[-1]):
            f = self.RBs[i](f)
        f = self.conv_tailf(f)
        refine_hsi = f + ref + coarse_hsi

        if self.flag:
            output = {"pred": refine_hsi, "de_lr_hsi": de_lr_hsi, "coarse_hsi": coarse_hsi}
        else:
            output = {"pred": refine_hsi, "coarse_hsi": coarse_hsi}
        return output


class HyperAE_ms(nn.Module):
    "嵌入AE和sr模块，使用self attention优化光谱"

    def __init__(self, config):
        super(HyperAE_ms, self).__init__()

        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.HR_size = config[config["train_dataset"]]["HR_size"]
        self.LR_size = config[config["train_dataset"]]["LR_size"]
        self.LFE = LFE(self.in_channels + 3)
        self.channels = config["AE"]["channels"]
        self.upscale_method = config["upscale_method"]
        if self.upscale_method == "sr":
            self.sr = MODELS[config["sr"]["sr_model"]](config)
        self.AE = AE(config)
        self.factor = config[config["train_dataset"]]["factor"]
        self.flag=config["train"]

        self.num_res_blocks = [0, 0, 0, 0]
        #self.outchannels = [128, 256, 512]
        self.outchannels = [64, 128, 256]
        self.res_scale = 1

        # LR_size at different level from AE
        self.lv1_size = int(self.LR_size / 2)
        self.lv2_size = int(self.LR_size / 4)
        self.lv3_size = int(self.LR_size / 8)

        #
        self.spectralattention1 = SpectralAttention(n_feats=self.channels[0])  # c=128
        self.spectralattention2 = SpectralAttention(n_feats=self.channels[1])  # c=256
        self.spectralattention3 = SpectralAttention(n_feats=self.channels[2])  # c=512

        self.RB1 = nn.ModuleList()
        for i in range(self.num_res_blocks[0]):
            self.RB1.append(ResBlock(in_channels=self.outchannels[0], out_channels=self.outchannels[0], res_scale=self.res_scale))

        self.RB2 = nn.ModuleList()
        for i in range(self.num_res_blocks[1]):
            self.RB2.append(ResBlock(in_channels=self.outchannels[1], out_channels=self.outchannels[1], res_scale=self.res_scale))

        self.RB3 = nn.ModuleList()
        for i in range(self.num_res_blocks[2]):
            self.RB3.append(ResBlock(in_channels=self.outchannels[2], out_channels=self.outchannels[2], res_scale=self.res_scale))

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks[-1]):
            self.RBs.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))

        self.conv_tail1 = nn.Conv2d(in_channels=self.outchannels[0], out_channels=self.outchannels[0], kernel_size=3, padding=1)
        self.conv_tail2 = nn.Conv2d(in_channels=self.outchannels[1], out_channels=self.outchannels[1], kernel_size=3, padding=1)
        self.conv_tail3 = nn.Conv2d(in_channels=self.outchannels[2], out_channels=self.outchannels[2], kernel_size=3, padding=1)
        self.final = nn.Conv2d(in_channels=sum(self.outchannels), out_channels=self.out_channels, kernel_size=1, padding=0)
        self.conv_tailf = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1)

        self.conv32 = nn.Conv2d(in_channels=self.outchannels[2], out_channels=2 * self.outchannels[2], kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(in_channels=self.outchannels[1], out_channels=2 * self.outchannels[1], kernel_size=3, padding=1)

        self.ps32 = nn.PixelShuffle(2)
        self.ps21 = nn.PixelShuffle(2)

    def forward(self, PAN, X_MS, MS_dhp=None):
        if self.upscale_method == "interpolate" and MS_dhp is None:
            coarse_hsi = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        if self.upscale_method == "DHP" and MS_dhp is not None:
            coarse_hsi = MS_dhp
        if self.upscale_method == "sr" and MS_dhp is None:
            coarse_hsi = self.sr(PAN, X_MS)  # b, c, H, W



            #PAN = PAN.unsqueeze(dim=1)
            b, c, h, w = X_MS.size()

        # 对X_MS进行自编码监督，压缩空间特征，提取光谱特征
        if self.flag:
            out, de_lr_hsi = self.AE(X_MS)  # en: b, c, h, w
        else:
            out = self.AE(X_MS)
            # 获取AE编码器前三层输出并reshape，每层输出空间尺度为H/2, H/4,H/8, 通道数分别为128，256，512
        en = []
        b, c1, hw = out[0].size()
        for i in range(3):
            entmp = out[i]  # b, c, hw
            h = int(h / 2)
            w = int(w / 2)
            en.append(entmp.view(b, c1, h, w))
            c1 = int(2 * c1)

        feats = torch.cat((coarse_hsi, PAN), dim=1)
        f1, f2, f3 = self.LFE(feats)  # 特征提取，三个空间尺度分别为H, H/2, H/4, 通道数分别为128，256，512
        #f111=f1
        t1, am1 = self.spectralattention1(en[0], f1)  # b, c, H, W
        t2, am2 = self.spectralattention2(en[1], f2)  # b, 2c, H/2, W/2
        t3, am3 = self.spectralattention3(en[2], f3)  # b, 4c, H/4, W/4

        # lv3, lr(L/8, W/8), hr(L/4, W/4) , t3--> b, 4c, H/4, W/4
        f3 = f3 + t3
        f3_res = f3
        for i in range(self.num_res_blocks[2]):
            f3 = self.RB3[i](f3)
        f3 = self.conv_tail3(f3)
        f3 = f3 + f3_res  # b, 4c, H/4, W/4

        # lv2, lr(L/4, W/4), hr(L/2, W/2) , t2--> b, 2c, H/2, W/2
        f22 = F.relu(self.ps32(self.conv32(f3)))  # b 2c H/2 W/2
        f2 = f2 + t2 + f22
        f2_res = f2
        for i in range(self.num_res_blocks[1]):
            f2 = self.RB2[i](f2)
        f2 = self.conv_tail2(f2)
        f2 = f2 + f2_res  # b 2c H/2 W/2

        # lv1, lr(L/2, W/2), hr(L, W) , t1--> b, c, H, W
        f11 = F.relu(self.ps21(self.conv21(f2)))  # b c H W
        f1 = f1 + t1 + f11
        f1_res = f1
        for i in range(self.num_res_blocks[0]):
            f1 = self.RB1[i](f1)
        f1 = self.conv_tail1(f1)
        f11 = f1 + f1_res

        f22 = F.interpolate(f2, scale_factor=2, mode='bicubic')
        f33 = F.interpolate(f3, scale_factor=4, mode='bicubic')

        f = torch.cat((f11, f22, f33), dim=1)
        f = self.final(f)
        ref = f
        for i in range(self.num_res_blocks[-1]):
            f = self.RBs[i](f)
        f = self.conv_tailf(f)
        refine_hsi = f + ref + coarse_hsi

        if self.flag:
            output = {"pred": refine_hsi, "de_lr_hsi": de_lr_hsi, "coarse_hsi": coarse_hsi}
        else:
            output = {"pred": refine_hsi, "coarse_hsi": coarse_hsi}
        return output


