# -*- coding：UTF-8 -*-
'''
@Project : HyperRefiner
@File ：sr1.py
@Author : Zerbo
@Date : 2022/10/31 18:31
'''


import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F
import sys
import os
import numpy as np
import cv2
import math


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


class SEBlock(nn.Module):
    """ Squeeze-and-excitation block """

    def __init__(self, channels, r=16):
        super(SEBlock, self).__init__()
        self.r = r
        self.squeeze = nn.Sequential(nn.Linear(channels, channels // self.r), nn.ReLU(), nn.Linear(channels // self.r, channels), nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.size()
        squeeze = torch.mean(x, dim=2)
        squeeze = torch.mean(squeeze, dim=2)
        squeeze = self.squeeze(squeeze).view(B, C, 1, 1)
        return torch.mul(x, squeeze)


class ECABlock(nn.Module):
    """ECA module"""

    def __init__(self, channels, b=1, gamma=2):
        super(ECABlock, self).__init__()
        # 自适应卷积核大小
        self.kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        if self.kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size, padding=(self.kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        """
        x: 生成通道权重的特征
        y: 用于注意力的特征,
        同一尺度下，将融合后的特征通道对齐到
        """
        b, c, h, w = x.size()
        z = self.avg_pool(x)

        # Two different branches of ECA module
        z = self.conv(z.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        z = self.sigmoid(z)
        return y * z.expand_as(y)  # 维度扩展


class SABlock(nn.Module):
    """ Spatial self-attention block """
    def __init__(self, in_channels, out_channels):
        super(SABlock, self).__init__()
        self.attention = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                                       nn.Sigmoid())
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)

    def forward(self, x, y):
        """
        x: 输入特征以得到权重
        y: 用于注意力的特征
        """
        attention_mask = self.attention(x)
        features = self.conv(y)
        return torch.mul(features, attention_mask)


class DualAttentionBlock(nn.Module):
    """ DualAttention. SA and CA for fusion of c&s """
    """ 空间特征和光谱特征在融合后做自注意力"""

    def __init__(self, in_channels, out_channels):
        super(DualAttentionBlock, self).__init__()
        # 输入输出通道为融合特征的通道
        self.in_channels = in_channels  # 2c
        self.out_channels = out_channels  # c

        self.CA = ECABlock(self.in_channels)
        self.SA = SABlock(self.in_channels, self.in_channels)
        self.aggregate1 = nn.Conv2d(in_channels=2 * self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1)
        self.aggregate2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, fused_feats):
        # fused_feats = self.FE(fused_feats)        #
        sa = self.SA(fused_feats, fused_feats)
        ca = self.CA(fused_feats, fused_feats)
        feats = torch.cat((sa, ca), dim=1)  # 4c
        feats = self.aggregate1(feats)  # 2c
        out = feats + fused_feats  # skip  2c
        out = self.aggregate2(out)  # c
        return out


class RefineAttentionBlock(nn.Module):
    """ RefineAttention. fuse to apply CA&SA """
    """ 利用融合后特征对输入的空间和光谱特征进行细化"""

    def __init__(self, in_channels, out_channels):
        super(RefineAttentionBlock, self).__init__()
        self.in_channels = in_channels  # c
        self.out_channels = out_channels  # c
        self.FE = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1),
                                nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1))
        self.CA = ECABlock(self.in_channels)
        # self.CA = FcaLayer(self.in_channels)
        self.SA = SABlock(self.in_channels, self.in_channels)  # self.aggregate = nn.Conv2d(3*self.in_channels, self.in_channels, kernel_size=3, padding=1)

    def forward(self, lr_feats, pan_feats, fused_feats):
        """
        lr_feats: 相同尺度下低分辨率高光谱特征
        pan_feats: 相同尺度下pan特征
        fused_feats: 该尺度下concat的特征, 通道数为c
        """
        # transform
        fused_feats = self.FE(fused_feats)
        # lr_feats = self.lr(lr_feats)
        # pan_feats = self.pan(pan_feats)
        # refine attention
        sa = self.SA(fused_feats, pan_feats)
        ca = self.CA(fused_feats, lr_feats)
        feats = sa + ca + fused_feats  # c
        # feats = torch.cat((sa, ca, fused_feats), dim=1)
        # feats = self.aggregate(feats)
        return feats


class DualAttentionBlock1(nn.Module):
    """ DualAttention. SA and CA for fusion of c&s """
    """ 空间特征和光谱特征在融合后做自注意力"""

    def __init__(self, in_channels, out_channels):
        super(DualAttentionBlock1, self).__init__()
        # 输入输出通道为融合特征的通道
        self.in_channels = in_channels  # 2c
        self.out_channels = out_channels  # c

        self.CA = ECABlock(self.in_channels)
        self.SA = SABlock(self.in_channels, self.in_channels)
        self.aggregate1 = nn.Conv2d(in_channels=2 * self.in_channels, out_channels=self.in_channels, kernel_size=1, padding=0)
        self.aggregate2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, fused_feats):
        # fused_feats = self.FE(fused_feats)        #
        sa = self.SA(fused_feats, fused_feats)
        ca = self.CA(fused_feats, fused_feats)
        feats = torch.cat((sa, ca), dim=1)  # 4c
        feats = self.aggregate1(feats)  # 2c
        out = feats + fused_feats  # skip  2c
        out = self.aggregate2(out)  # c
        return out


class RefineAttentionBlock1(nn.Module):
    """ RefineAttention. fuse to apply CA&SA """
    """ 利用融合后特征对输入的空间和光谱特征进行细化"""

    def __init__(self, in_channels, out_channels):
        super(RefineAttentionBlock2, self).__init__()
        self.in_channels = in_channels  # c
        self.out_channels = out_channels  # c
        self.FE = nn.Sequential(nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1),
                                nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=3, padding=1))
        self.CA = ECABlock(self.in_channels)
        self.SA = SABlock(self.in_channels, self.in_channels)
        self.aggregate = nn.Conv2d(3 * self.in_channels, 2 * self.in_channels, kernel_size=1, padding=0)

    def forward(self, lr_feats, pan_feats, fused_feats):
        """
        lr_feats: 相同尺度下低分辨率高光谱特征   c
        pan_feats: 相同尺度下pan特征  c
        fused_feats: 该尺度下concat的特征, 通道数为c
        """
        # transform
        fused_feats = self.FE(fused_feats)
        # lr_feats = self.lr(lr_feats)
        # pan_feats = self.pan(pan_feats)
        # refine attention
        sa = self.SA(fused_feats, pan_feats)
        ca = self.CA(fused_feats, lr_feats)
        # feats = sa + ca + fused_feats  # c
        feats = torch.cat((sa, ca, fused_feats), dim=1)
        feats = self.aggregate(feats)  # 2c
        return feats


class DualAttentionBlock2(nn.Module):
    """ DualAttention. SA and CA for fusion of c&s """
    """ 空间特征和光谱特征在融合后做自注意力"""
    def __init__(self, in_channels, out_channels):
        super(DualAttentionBlock2, self).__init__()
        #输入输出通道为融合特征的通道
        self.in_channels = in_channels   #2c
        self.out_channels = out_channels  #c

        self.CA = ECABlock(self.in_channels)
        self.SA = SABlock(self.in_channels, self.in_channels)
        #self.aggregate1 = nn.Conv2d(in_channels=2*self.in_channels, out_channels=self.in_channels, kernel_size=1, padding=0)
        self.aggregate2 = nn.Conv2d(in_channels=self.in_channels, out_channels=2*self.out_channels, kernel_size=3, padding=1)

    def forward(self, fused_feats):
        #fused_feats = self.FE(fused_feats)        #
        sa = self.SA(fused_feats, fused_feats)
        ca = self.CA(fused_feats, fused_feats)

        out = sa + ca + fused_feats           #skip  2c
        out = self.aggregate2(out)          #c
        return out


class RefineAttentionBlock2(nn.Module):
    """ RefineAttention. fuse to apply CA&SA """
    """ 利用融合后特征对输入的空间和光谱特征进行细化"""
    def __init__(self, in_channels, out_channels):
        super(RefineAttentionBlock2, self).__init__()
        self.in_channels = in_channels  #c
        self.out_channels = out_channels  #c
        self.FE = nn.Conv2d(in_channels=2*self.in_channels, out_channels=self.in_channels, kernel_size=1, padding=0)

        self.CA = ECABlock(self.in_channels)
        self.SA = SABlock(self.in_channels, self.in_channels)
        self.aggregate = nn.Conv2d(3*self.in_channels, 2*self.in_channels, kernel_size=1, padding=0)

    def forward(self, lr_feats, pan_feats, fused_feats):
        """
        lr_feats: 相同尺度下低分辨率高光谱特征   c
        pan_feats: 相同尺度下pan特征  c
        fused_feats: 该尺度下concat的特征, 通道数为2c
        """
        #transform
        fused_feats = self.FE(fused_feats)  #c
        #lr_fe
        #pan_feats = self.pan(pan_feats)
        #refine attention
        sa = self.SA(fused_feats, pan_feats)
        ca = self.CA(fused_feats, lr_feats)
        #feats = sa+ca+fused_feats          #c
        feats = torch.cat((sa, ca, fused_feats), dim=1)
        feats = self.aggregate(feats)      #2c
        return feats


class DRModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_rbs):
        super(DRModule, self).__init__()
        self.num_rbs = num_rbs
        self.DAB = DualAttentionBlock(2 * in_channels, out_channels)
        self.RAB = RefineAttentionBlock(in_channels, out_channels)
        self.RBs = nn.ModuleList()
        for i in range(self.num_rbs):
            self.RBs.append(ResBlock(in_channels=out_channels, out_channels=out_channels))

    def forward(self, pan_feats, lr_feats, fuse=None):
        """
        fuse: 上一级输出的refine结果，通道数c
        """
        fused = torch.cat((pan_feats, lr_feats), dim=1)
        refine = self.DAB(fused)  # 128  H  W
        if fuse is None:
            refine = self.RAB(pan_feats, lr_feats, refine)
        else:
            refine = refine + fuse
            refine = self.RAB(pan_feats, lr_feats, refine)
        refine1 = refine
        for i in range(self.num_rbs):
            refine = self.RBs[i](refine)
        refine = refine1 + refine

        return refine


class DRModule1(nn.Module):
    def __init__(self, in_channels, out_channels, num_rbs):
        super(DRModule1, self).__init__()
        self.num_rbs = num_rbs
        self.DAB = DualAttentionBlock(2 * in_channels, out_channels)
        self.RAB = RefineAttentionBlock(in_channels, out_channels)
        self.RBs = nn.ModuleList()
        for i in range(self.num_rbs):
            self.RBs.append(ResBlock(in_channels=out_channels, out_channels=out_channels))

    def forward(self, pan_feats, lr_feats):
        """
        fuse: 上一级输出的refine结果，通道数c
        """
        fused = torch.cat((pan_feats, lr_feats), dim=1)
        refine = self.DAB(fused)  # 128  H  W

        refine1 = refine
        for i in range(self.num_rbs):
            refine = self.RBs[i](refine)
        refine = refine1 + refine
        refine = self.RAB(pan_feats, lr_feats, refine)

        return refine


class DRModule2(nn.Module):
    def __init__(self, in_channels, out_channels, num_rbs):
        super(DRModule2, self).__init__()
        self.num_rbs = num_rbs
        self.DAB = DualAttentionBlock2(2 * in_channels, out_channels)
        self.RAB = RefineAttentionBlock2(in_channels, out_channels)
        self.RBs = nn.ModuleList()
        for i in range(self.num_rbs):
            self.RBs.append(ResBlock(in_channels=2 * out_channels, out_channels=2 * out_channels))

    def forward(self, pan_feats, lr_feats, refine=None):
        """
        refine: 上一级输出的refine结果，通道数2c
        """
        fused = torch.cat((pan_feats, lr_feats), dim=1)
        if refine is not None:
            fused = fused + refine
        refine = self.DAB(fused)
        refine = self.RAB(pan_feats, lr_feats, refine)
        refine1 = refine
        for i in range(self.num_rbs):
            refine = self.RBs[i](refine)
        refine = refine1 + refine

        return refine


class DRModule3(nn.Module):
    def __init__(self, in_channels, out_channels, num_rbs):
        super(DRModule3, self).__init__()
        self.num_rbs = num_rbs
        self.DAB = DualAttentionBlock2(2 * in_channels, out_channels)
        self.RAB = RefineAttentionBlock2(in_channels, out_channels)
        self.RBs = nn.ModuleList()
        for i in range(self.num_rbs):
            self.RBs.append(ResBlock(in_channels=2 * out_channels, out_channels=2 * out_channels))

    def forward(self, pan_feats, lr_feats, refine=None):
        """
        refine: 上一级输出的refine结果，通道数2c
        """
        fused = torch.cat((pan_feats, lr_feats), dim=1)
        if refine is not None:
            fused = fused + refine
        refine = self.DAB(fused)
        refine = self.RAB(pan_feats, lr_feats, refine)  # 2c
        refine1 = refine + fused
        for i in range(self.num_rbs):
            refine = self.RBs[i](refine)
        refine = refine1 + refine

        return refine


# 初步重建模块，实现较高质量的高光谱影像上采样
class Coarse_sr1(nn.Module):
    def __init__(self, config):
        super(Coarse_sr1, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]

        self.outchannels = [64, 128, 256]
        self.num_res_blocks = [4, 4, 4]
        self.res_scale = 1

        self.SFE = LFE(in_channels=1)
        self.CFE = LFE(in_channels=self.in_channels)

        self.DAB1 = DualAttentionBlock(2 * self.outchannels[0], self.outchannels[0])
        self.RAB1 = RefineAttentionBlock(self.outchannels[0], self.outchannels[0])

        self.DAB2 = DualAttentionBlock(2 * self.outchannels[1], self.outchannels[1])
        self.RAB2 = RefineAttentionBlock(self.outchannels[1], self.outchannels[1])

        self.DAB3 = DualAttentionBlock(2 * self.outchannels[2], self.outchannels[2])
        self.RAB3 = RefineAttentionBlock(self.outchannels[2], self.outchannels[2])

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
        for i in range(4):
            self.RBs.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))

        self.conv32 = nn.Conv2d(in_channels=self.outchannels[2], out_channels=2 * self.outchannels[2], kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(in_channels=self.outchannels[1], out_channels=2 * self.outchannels[1], kernel_size=3, padding=1)

        self.conv_tail1 = nn.Conv2d(in_channels=self.outchannels[0], out_channels=self.outchannels[0], kernel_size=3, padding=1)
        self.conv_tail2 = nn.Conv2d(in_channels=self.outchannels[1], out_channels=self.outchannels[1], kernel_size=3, padding=1)
        self.conv_tail3 = nn.Conv2d(in_channels=self.outchannels[2], out_channels=self.outchannels[2], kernel_size=3, padding=1)
        self.final = nn.Conv2d(in_channels=sum(self.outchannels), out_channels=self.out_channels, kernel_size=3, padding=1)
        self.conv_tailf = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1)

        self.ps32 = nn.PixelShuffle(2)
        self.ps21 = nn.PixelShuffle(2)
        self.up_conv31 = nn.ConvTranspose2d(in_channels=self.outchannels[2], out_channels=self.in_channels, kernel_size=3, stride=4, output_padding=1)
        self.up_conv21 = nn.ConvTranspose2d(in_channels=self.outchannels[1], out_channels=self.in_channels, kernel_size=3, stride=2, padding=1,
                                            output_padding=1)

    def forward(self, X_MS, X_PAN):
        # 对低分辨率高光谱进行简单上采样至高分辨大小
        X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        X_PAN = X_PAN.unsqueeze(dim=1)

        # lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
        S_lv1, S_lv2, S_lv3 = self.SFE(X_PAN)
        C_lv1, C_lv2, C_lv3 = self.CFE(X_MS_UP)

        # lv3=(L/4, W/4)
        fused3 = torch.cat((S_lv3, C_lv3), dim=1)
        refine3 = self.DAB3(fused3)
        refine33 = refine3
        for i in range(self.num_res_blocks[2]):
            refine3 = self.RB3[i](refine3)  # c=256
        refine3 = self.conv_tail3(refine3)
        refine3 = refine33 + refine3
        refine3 = self.RAB3(C_lv3, S_lv3, refine3)

        # lv2 = (L/2)x(w/2)
        # refine22 = F.relu(self.ps32(self.conv32(refine3)))
        fused2 = torch.cat((S_lv2, C_lv2), dim=1)
        refine2 = self.DAB2(fused2)
        # refine2 = refine22 + refine2
        refine22 = refine2
        for i in range(self.num_res_blocks[1]):
            refine2 = self.RB2[i](refine2)
        refine2 = self.conv_tail2(refine2)
        refine2 = refine2 + refine22
        refine2 = self.RAB2(C_lv2, S_lv2, refine2)

        # lv1 = LxW
        # refine11 = F.relu(self.ps21(self.conv21(refine2)))
        fused1 = torch.cat((S_lv1, C_lv1), dim=1)
        refine1 = self.DAB1(fused1)
        # refine1 = refine11 + refine1
        refine11 = refine1
        for i in range(self.num_res_blocks[0]):
            refine1 = self.RB1[i](refine1)
        refine1 = self.conv_tail1(refine1)
        refine1 = refine1 + refine11
        refine11 = self.RAB1(C_lv1, S_lv1, refine1)

        refine22 = F.interpolate(refine2, scale_factor=2, mode='bicubic')
        refine33 = F.interpolate(refine3, scale_factor=4, mode='bicubic')

        refine = torch.cat((refine11, refine22, refine33), dim=1)
        refine = self.final(refine)
        ref = refine
        for i in range(4):
            refine = self.RBs[i](refine)
        refine = self.conv_tailf(refine)
        refine = refine + ref + X_MS_UP

        ref3 = self.up_conv31(refine3)
        ref2 = self.up_conv21(refine2)
        output = {"pred": refine, "ref2": refine2, "ref3": refine3}
        return output


class Coarse_sr2(nn.Module):
    def __init__(self, config):
        super(Coarse_sr2, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]

        self.outchannels = [64, 64, 64]
        self.num_res_blocks = [4, 2, 1]
        self.res_scale = 1

        self.SFE = LFE(in_channels=1)
        self.CFE = LFE(in_channels=self.in_channels)

        self.DAB1 = DualAttentionBlock(2 * self.outchannels[0], self.outchannels[0])
        self.RAB1 = RefineAttentionBlock(self.outchannels[0], self.outchannels[0])

        self.DAB2 = DualAttentionBlock(2 * self.outchannels[1], self.outchannels[1])
        self.RAB2 = RefineAttentionBlock(self.outchannels[1], self.outchannels[1])

        self.DAB3 = DualAttentionBlock(2 * self.outchannels[2], self.outchannels[2])
        self.RAB3 = RefineAttentionBlock(self.outchannels[2], self.outchannels[2])

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
        for i in range(2):
            self.RBs.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))

        self.conv32 = nn.Conv2d(in_channels=self.outchannels[2], out_channels=2 * self.outchannels[2], kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(in_channels=self.outchannels[1], out_channels=2 * self.outchannels[1], kernel_size=3, padding=1)

        self.conv_tail1 = nn.Conv2d(in_channels=self.outchannels[0], out_channels=self.outchannels[0], kernel_size=3, padding=1)
        self.conv_tail2 = nn.Conv2d(in_channels=self.outchannels[1], out_channels=self.outchannels[1], kernel_size=3, padding=1)
        self.conv_tail3 = nn.Conv2d(in_channels=self.outchannels[2], out_channels=self.outchannels[2], kernel_size=3, padding=1)
        self.final = nn.Conv2d(in_channels=sum(self.outchannels), out_channels=self.out_channels, kernel_size=3, padding=1)
        self.conv_tailf = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1)

        self.ps32 = nn.PixelShuffle(2)
        self.ps21 = nn.PixelShuffle(2)

    def forward(self, X_MS, X_PAN):
        # 对低分辨率高光谱进行简单上采样至高分辨大小
        X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        X_PAN = X_PAN.unsqueeze(dim=1)

        # lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
        S_lv1, S_lv2, S_lv3 = self.SFE(X_PAN)
        C_lv1, C_lv2, C_lv3 = self.CFE(X_MS_UP)

        # lv3=(L/4, W/4)
        fused3 = torch.cat((S_lv3, C_lv3), dim=1)
        refine3 = self.DAB3(fused3)
        # for i in range(self.num_res_blocks[2]):
        #    refine3 = self.RB3[i](refine3)         #c=256
        # refine3 = self.conv_tail3(refine3)
        # refine3 = refine33 + refine3
        refine3 = self.RAB3(C_lv3, S_lv3, refine3)
        refine33 = refine3
        for i in range(self.num_res_blocks[2]):
            refine3 = self.RB3[i](refine3)  # c=256
        refine3 = self.conv_tail3(refine3)
        refine3 = refine33 + refine3
        # refine3 = refine33 + refine3

        # lv2 = (L/2)x(w/2)
        refine22 = F.relu(self.ps32(self.conv32(refine3)))
        fused2 = torch.cat((S_lv2, C_lv2), dim=1)
        refine2 = self.DAB2(fused2)
        refine2 = refine22 + refine2
        refine2 = self.RAB2(C_lv2, S_lv2, refine2)
        refine22 = refine2
        for i in range(self.num_res_blocks[1]):
            refine2 = self.RB2[i](refine2)
        refine2 = self.conv_tail2(refine2)
        refine2 = refine2 + refine22

        # lv1 = LxW
        refine11 = F.relu(self.ps21(self.conv21(refine2)))
        fused1 = torch.cat((S_lv1, C_lv1), dim=1)
        refine1 = self.DAB1(fused1)
        refine1 = refine11 + refine1
        refine1 = self.RAB1(C_lv1, S_lv1, refine1)
        refine11 = refine1
        for i in range(self.num_res_blocks[0]):
            refine1 = self.RB1[i](refine1)
        refine1 = self.conv_tail1(refine1)
        refine1 = refine1 + refine11

        refine22 = F.interpolate(refine22, scale_factor=2, mode='bicubic')
        refine33 = F.interpolate(refine33, scale_factor=4, mode='bicubic')

        refine = torch.cat((refine1, refine22, refine33), dim=1)
        refine = self.final(refine)
        ref = refine
        for i in range(2):
            refine = self.RBs[i](refine)
        refine = self.conv_tailf(refine)
        refine = refine + ref

        output = {"pred": refine, "ref2": refine22, "ref3": refine33}
        return output


class Coarse_sr3(nn.Module):
    def __init__(self, config):
        super(Coarse_sr3, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]

        self.outchannels = [64, 128, 256]
        self.num_res_blocks = [2, 2, 1]
        self.res_scale = 1

        self.SFE = LFE(in_channels=1)
        self.CFE = LFE(in_channels=self.in_channels)

        self.DAB1 = DualAttentionBlock(2 * self.outchannels[0], self.outchannels[0])
        self.RAB1 = RefineAttentionBlock(self.outchannels[0], self.outchannels[0])

        self.DAB2 = DualAttentionBlock(2 * self.outchannels[1], self.outchannels[1])
        self.RAB2 = RefineAttentionBlock(self.outchannels[1], self.outchannels[1])

        self.DAB3 = DualAttentionBlock(2 * self.outchannels[2], self.outchannels[2])
        self.RAB3 = RefineAttentionBlock(self.outchannels[2], self.outchannels[2])

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
        for i in range(1):
            self.RBs.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))

        self.conv32 = nn.Conv2d(in_channels=self.outchannels[2], out_channels=2 * self.outchannels[2], kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(in_channels=self.outchannels[1], out_channels=2 * self.outchannels[1], kernel_size=3, padding=1)

        self.conv_tail1 = nn.Conv2d(in_channels=self.outchannels[0], out_channels=self.outchannels[0], kernel_size=3, padding=1)
        self.conv_tail2 = nn.Conv2d(in_channels=self.outchannels[1], out_channels=self.outchannels[1], kernel_size=3, padding=1)
        self.conv_tail3 = nn.Conv2d(in_channels=self.outchannels[2], out_channels=self.outchannels[2], kernel_size=3, padding=1)
        self.final = nn.Conv2d(in_channels=sum(self.outchannels), out_channels=self.out_channels, kernel_size=3, padding=1)
        self.conv_tailf = nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, padding=1)

        self.ps32 = nn.PixelShuffle(2)
        self.ps21 = nn.PixelShuffle(2)

    def forward(self, X_MS, X_PAN):
        # 对低分辨率高光谱进行简单上采样至高分辨大小
        X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        X_PAN = X_PAN.unsqueeze(dim=1)

        # lv1 = LxW, lv2 = (L/2)x(w/2), lv3=(L/4, W/4)
        S_lv1, S_lv2, S_lv3 = self.SFE(X_PAN)
        C_lv1, C_lv2, C_lv3 = self.CFE(X_MS_UP)

        # lv3=(L/4, W/4)
        fused3 = torch.cat((S_lv3, C_lv3), dim=1)
        refine3 = self.DAB3(fused3)
        # for i in range(self.num_res_blocks[2]):
        #    refine3 = self.RB3[i](refine3)         #c=256
        # refine3 = self.conv_tail3(refine3)
        # refine3 = refine33 + refine3
        refine3 = self.RAB3(C_lv3, S_lv3, refine3)
        refine33 = refine3
        for i in range(self.num_res_blocks[2]):
            refine3 = self.RB3[i](refine3)  # c=256
        refine3 = self.conv_tail3(refine3)
        refine3 = refine33 + refine3
        # refine3 = refine33 + refine3

        # lv2 = (L/2)x(w/2)
        refine22 = F.relu(self.ps32(self.conv32(refine3)))
        fused2 = torch.cat((S_lv2, C_lv2), dim=1)
        refine2 = self.DAB2(fused2)
        refine2 = refine22 + refine2
        refine2 = self.RAB2(C_lv2, S_lv2, refine2)
        refine22 = refine2
        for i in range(self.num_res_blocks[1]):
            refine2 = self.RB2[i](refine2)
        refine2 = self.conv_tail2(refine2)
        refine2 = refine2 + refine22

        # lv1 = LxW
        refine11 = F.relu(self.ps21(self.conv21(refine2)))
        fused1 = torch.cat((S_lv1, C_lv1), dim=1)
        refine1 = self.DAB1(fused1)
        refine1 = refine11 + refine1
        refine1 = self.RAB1(C_lv1, S_lv1, refine1)
        refine11 = refine1
        for i in range(self.num_res_blocks[0]):
            refine1 = self.RB1[i](refine1)
        refine1 = self.conv_tail1(refine1)
        refine1 = refine1 + refine11

        refine22 = F.interpolate(refine22, scale_factor=2, mode='bicubic')
        refine33 = F.interpolate(refine33, scale_factor=4, mode='bicubic')

        refine = torch.cat((refine1, refine22, refine33), dim=1)
        refine = self.final(refine)
        ref = refine
        for i in range(1):
            refine = self.RBs[i](refine)
        refine = self.conv_tailf(refine)
        refine = refine + ref

        output = {"pred": refine, "ref2": refine22, "ref3": refine33}
        return output


class Coarse_sr4(nn.Module):
    def __init__(self, config):
        """
        级联结构 version2
        """
        super(Coarse_sr4, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]

        self.outchannels = [64, 64, 64, 64]
        self.num_blocks = len(self.outchannels) - 1
        self.num_res_blocks = [4, 4, 4]
        self.res_scale = 1

        self.pan = nn.Conv2d(1, self.outchannels[0], kernel_size=3, padding=1, stride=1)
        self.lr_feats = nn.Conv2d(self.in_channels, self.outchannels[0], kernel_size=3, padding=1, stride=1)

        self.Features = nn.ModuleList()
        self.DRB = nn.ModuleList()

        for i in range(self.num_blocks):
            self.Features.append(nn.Sequential(nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1),
                                               nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1)))
            self.DRB.append(DRModule(self.outchannels[i], self.outchannels[i + 1], self.num_res_blocks[i]))

        self.conv_tail = nn.Conv2d(in_channels=self.outchannels[-1], out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, X_MS, X_PAN):
        # channel and spatial formulated
        X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        X_PAN = X_PAN.unsqueeze(dim=1)

        # head
        pan = self.pan(X_PAN)
        lr_feats = self.lr_feats(X_MS_UP)

        # DRBlocks
        for i in range(self.num_blocks):
            pan = self.Features[i](pan)
            lr_feats = self.Features[i](lr_feats)

            if i == 0:  # 第一步，输入的只有空间、光谱特征的拼接
                refine = self.DRB[i](pan, lr_feats, fuse=None)  # c
            else:  # 后续，将上一步的refine结果加入
                refine = self.DRB[i](pan, lr_feats, fuse=refine)

        refine = self.conv_tail(refine)
        refine = refine + X_MS_UP
        output = {"pred": refine}
        return output


class Coarse_sr5(nn.Module):
    def __init__(self, config):
        """
        级联结构 version2
        """
        super(Coarse_sr5, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]

        self.outchannels = [64, 64, 64, 64]
        self.num_blocks = len(self.outchannels) - 1
        self.num_res_blocks = [4, 4, 4]
        self.res_scale = 1

        self.pan = nn.Conv2d(1, self.outchannels[0], kernel_size=3, padding=1, stride=1)
        self.lr_feats = nn.Conv2d(self.in_channels, self.outchannels[0], kernel_size=3, padding=1, stride=1)

        self.Features = nn.ModuleList()
        self.DRB = nn.ModuleList()

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks[-1]):
            self.RBs.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        for i in range(self.num_blocks):
            self.Features.append(nn.Sequential(nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1),
                                               nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1)))
            self.DRB.append(DRModule1(self.outchannels[i], self.outchannels[i + 1], self.num_res_blocks[i]))

        self.conv_tail = nn.Conv2d(in_channels=3 * self.outchannels[-1], out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, X_MS, X_PAN):
        # channel and spatial formulated
        X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        X_PAN = X_PAN.unsqueeze(dim=1)

        # head
        pan = self.pan(X_PAN)
        lr_feats = self.lr_feats(X_MS_UP)
        ref = []

        # DRBlocks
        for i in range(self.num_blocks):
            pan = self.Features[i](pan)
            lr_feats = self.Features[i](lr_feats)
            refine = self.DRB[i](pan, lr_feats)  # c
            ref.append(refine)

        refine = torch.cat((ref[0], ref[1], ref[2]), dim=1)
        refine = self.conv_tail(refine)

        ref = refine
        for i in range(self.num_res_blocks[-1]):
            refine = self.RBs[i](refine)
        refine = refine + ref

        output = {"pred": refine}

        return output


class Coarse_sr6(nn.Module):
    def __init__(self, config):
        """
        级联结构 version2
        """
        super(Coarse_sr6, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]

        self.outchannels = [64, 64, 64, 64]
        self.num_blocks = len(self.outchannels) - 1
        self.num_res_blocks = [0, 0, 0, 0]
        self.res_scale = 1

        self.pan = nn.Conv2d(1, self.outchannels[0], kernel_size=3, padding=1, stride=1)
        self.lr_feats = nn.Conv2d(self.in_channels, self.outchannels[0], kernel_size=3, padding=1, stride=1)

        self.Features = nn.ModuleList()
        self.DRB = nn.ModuleList()

        for i in range(self.num_blocks):
            self.Features.append(nn.Sequential(nn.Conv2d(self.outchannels[i], 2*self.outchannels[i], kernel_size=3, padding=1, stride=1),
                                               nn.Conv2d(2*self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1)))
            self.DRB.append(DRModule2(self.outchannels[i], self.outchannels[i + 1], self.num_res_blocks[i]))

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks[-1]):
            self.RBs.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.conv_tail = nn.Conv2d(in_channels=2 * self.outchannels[-1], out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self,  X_MS, X_PAN):
        # channel and spatial formulated
        X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        X_PAN = X_PAN.unsqueeze(dim=1)

        # head
        pan = self.pan(X_PAN)
        lr_feats = self.lr_feats(X_MS_UP)
        ref = []

        # DRBlocks
        for i in range(self.num_blocks):
            pan = self.Features[i](pan)
            lr_feats = self.Features[i](lr_feats)
            refine = self.DRB[i](pan, lr_feats)  # 2c
            ref.append(refine)
        refine = ref[0] + ref[1] + ref[2]
        refine = self.conv_tail(refine)  # c

        ref = refine
        for i in range(self.num_res_blocks[-1]):
            refine = self.RBs[i](refine)
        refine = refine + ref + X_MS_UP

        output = {"pred": refine}
        return output
        #return refine


class Coarse_sr61(nn.Module):
    def __init__(self, config):
        """
        级联结构 version2
        """
        super(Coarse_sr61, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]

        self.outchannels = [64, 64, 64, 64]
        self.num_blocks = len(self.outchannels)-1
        self.num_res_blocks = [0, 0, 0, 0]
        self.res_scale = 1

        self.pan = nn.Conv2d(1, self.outchannels[0], kernel_size=3, padding=1, stride=1)
        self.lr_feats = nn.Conv2d(self.in_channels, self.outchannels[0], kernel_size=3, padding=1, stride=1)

        self.Features = nn.ModuleList()
        self.DRB = nn.ModuleList()

        for i in range(self.num_blocks):
            self.Features.append(nn.Sequential(
                    nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1),
                    nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1)
                    ))
            self.DRB.append(DRModule2(self.outchannels[i], self.outchannels[i+1], self.num_res_blocks[i]))

        #self.RBs = nn.ModuleList()
        #for i in range(self.num_res_blocks[-1]):
            #self.RBs.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.conv_tail = nn.Conv2d(in_channels=2*self.outchannels[-1], out_channels=self.out_channels, kernel_size=3, padding=1)



    def forward(self,  X_PAN, X_MS):
        #channel and spatial formulated
        X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        X_PAN = X_PAN.unsqueeze(dim=1)

        #head
        pan = self.pan(X_PAN)
        lr_feats = self.lr_feats(X_MS_UP)
        ref=[]

        #DRBlocks
        for i in range(self.num_blocks):
            pan = self.Features[i](pan)
            lr_feats = self.Features[i](lr_feats)
            refine = self.DRB[i](pan, lr_feats)  #2c
            ref.append(refine)
        refine = ref[0]+ref[1]+ref[2]
        refine = self.conv_tail(refine)          #c

        #ref = refine
        #for i in range(self.num_res_blocks[-1]):
            #refine = self.RBs[i](refine)
        refine = refine  + X_MS_UP

        #output = {"pred": refine}
        return refine
        #return output

class Coarse_sr61_ms(nn.Module):
    def __init__(self, config):
        """
        级联结构 version2
        """
        super(Coarse_sr61_ms, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]

        self.outchannels = [64, 64, 64, 64]
        self.num_blocks = len(self.outchannels)-1
        self.num_res_blocks = [0, 0, 0, 0]
        self.res_scale = 1

        self.pan = nn.Conv2d(3, self.outchannels[0], kernel_size=3, padding=1, stride=1)
        self.lr_feats = nn.Conv2d(self.in_channels, self.outchannels[0], kernel_size=3, padding=1, stride=1)

        self.Features = nn.ModuleList()
        self.DRB = nn.ModuleList()

        for i in range(self.num_blocks):
            self.Features.append(nn.Sequential(
                    nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1),
                    nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1)
                    ))
            self.DRB.append(DRModule2(self.outchannels[i], self.outchannels[i+1], self.num_res_blocks[i]))

        #self.RBs = nn.ModuleList()
        #for i in range(self.num_res_blocks[-1]):
            #self.RBs.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.conv_tail = nn.Conv2d(in_channels=2*self.outchannels[-1], out_channels=self.out_channels, kernel_size=3, padding=1)



    def forward(self,  X_MS,  X_PAN):
        #channel and spatial formulated
        X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        #X_PAN = X_PAN.unsqueeze(dim=1)

        #head
        pan = self.pan(X_PAN)
        lr_feats = self.lr_feats(X_MS_UP)
        ref=[]

        #DRBlocks
        for i in range(self.num_blocks):
            pan = self.Features[i](pan)
            lr_feats = self.Features[i](lr_feats)
            refine = self.DRB[i](pan, lr_feats)  #2c
            ref.append(refine)
        refine = ref[0]+ref[1]+ref[2]
        refine = self.conv_tail(refine)          #c

        #ref = refine
        #for i in range(self.num_res_blocks[-1]):
            #refine = self.RBs[i](refine)
        refine = refine  + X_MS_UP

        output = {"pred": refine}
        #return refine
        return output
class Coarse_sr7(nn.Module):
    def __init__(self, config):
        """
        级联结构 version2
        """
        super(Coarse_sr7, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]

        self.outchannels = [64, 64, 64, 64]
        self.num_blocks = len(self.outchannels) - 1
        self.num_res_blocks = [2, 2, 2, 2]
        self.res_scale = 1

        self.pan = nn.Conv2d(1, self.outchannels[0], kernel_size=3, padding=1, stride=1)
        self.lr_feats = nn.Conv2d(self.in_channels, self.outchannels[0], kernel_size=3, padding=1, stride=1)

        # 此处对空谱用不同参数的提取器
        self.SF = nn.ModuleList()
        self.CF = nn.ModuleList()
        self.DRB = nn.ModuleList()

        for i in range(self.num_blocks):
            self.SF.append(nn.Sequential(nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1),
                                         nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1)))
            self.CF.append(nn.Sequential(nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1),
                                         nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1)))
            self.DRB.append(DRModule2(self.outchannels[i], self.outchannels[i + 1], self.num_res_blocks[i]))

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks[-1]):
            self.RBs.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.conv_tail = nn.Conv2d(in_channels=2 * self.outchannels[-1], out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, X_MS, X_PAN):
        # channel and spatial formulated
        X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        X_PAN = X_PAN.unsqueeze(dim=1)

        # head
        pan = self.pan(X_PAN)
        lr_feats = self.lr_feats(X_MS_UP)
        ref = []

        # DRBlocks
        for i in range(self.num_blocks):
            pan = self.SF[i](pan)
            lr_feats = self.CF[i](lr_feats)
            refine = self.DRB[i](pan, lr_feats)  # 2c
            ref.append(refine)
        refine = ref[0] + ref[1] + ref[2]
        refine = self.conv_tail(refine)  # c

        ref = refine
        for i in range(self.num_res_blocks[-1]):
            refine = self.RBs[i](refine)
        refine = refine + ref

        output = {"pred": refine}
        return output


class Coarse_sr8(nn.Module):
    def __init__(self, config):
        """
        级联结构 version2 残差版
        """
        super(Coarse_sr8, self).__init__()
        self.is_DHP_MS = config["is_DHP_MS"]
        self.in_channels = config[config["train_dataset"]]["spectral_bands"]
        self.out_channels = config[config["train_dataset"]]["spectral_bands"]
        self.factor = config[config["train_dataset"]]["factor"]

        self.outchannels = [64, 64, 64, 64]
        self.num_blocks = len(self.outchannels) - 1
        self.num_res_blocks = [0, 0, 0, 0]
        self.res_scale = 1

        self.pan = nn.Conv2d(1, self.outchannels[0], kernel_size=3, padding=1, stride=1)
        self.lr_feats = nn.Conv2d(self.in_channels, self.outchannels[0], kernel_size=3, padding=1, stride=1)

        # 此处对空谱用不同参数的提取器
        self.SF = nn.ModuleList()
        self.CF = nn.ModuleList()
        self.DRB = nn.ModuleList()

        for i in range(self.num_blocks):
            self.SF.append(
                nn.Sequential(nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(self.outchannels[i]),
                              nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1),
                              nn.BatchNorm2d(self.outchannels[i]), nn.LeakyReLU(negative_slope=0.2)))
            self.CF.append(
                nn.Sequential(nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(self.outchannels[i]),
                              nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(self.outchannels[i], self.outchannels[i], kernel_size=3, padding=1, stride=1),
                              nn.BatchNorm2d(self.outchannels[i]), nn.LeakyReLU(negative_slope=0.2)))
            self.DRB.append(DRModule3(self.outchannels[i], self.outchannels[i + 1], self.num_res_blocks[i]))

        self.RBs = nn.ModuleList()
        for i in range(self.num_res_blocks[-1]):
            self.RBs.append(ResBlock(in_channels=self.out_channels, out_channels=self.out_channels, res_scale=self.res_scale))
        self.conv_tail = nn.Conv2d(in_channels=2 * self.outchannels[-1], out_channels=self.out_channels, kernel_size=3, padding=1)

    def forward(self, X_MS, X_PAN):
        # channel and spatial formulated
        X_MS_UP = F.interpolate(X_MS, scale_factor=(self.factor, self.factor), mode='bilinear')
        X_PAN = X_PAN.unsqueeze(dim=1)

        # head
        pan = self.pan(X_PAN)
        lr_feats = self.lr_feats(X_MS_UP)
        # ref=[]

        # DRBlocks
        for i in range(self.num_blocks):
            pan = self.SF[i](pan)
            lr_feats = self.CF[i](lr_feats)
            refine = self.DRB[i](pan, lr_feats)  # 2c  # ref.append(refine)
        # refine = ref[0]+ref[1]+ref[2]
        # refine = ref[2]
        refine = self.conv_tail(refine)  # c
        ref = refine
        for i in range(self.num_res_blocks[-1]):
            refine = self.RBs[i](refine)
        refine = refine + ref
        # refine = refine + X_MS_UP
        output = {"pred": refine}
        return output


class LFE(nn.Module):
    def __init__(self, in_channels):
        super(LFE, self).__init__()

        # Define number of input channels
        self.in_channels = in_channels
        self.filters = 128

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
