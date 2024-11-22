from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from .submodule import *
import math
import gc
import time
import timm

class SubModule(nn.Module):
    def __init__(self):
        super(SubModule, self).__init__()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Feature(SubModule):
    def __init__(self):
        super(Feature, self).__init__()
        pretrained =  True
        model = timm.create_model('mobilenetv2_100', pretrained=pretrained, features_only=True)
        layers = [1,2,3,5,6]
        chans = [16, 24, 32, 96, 160]
        self.conv_stem = model.conv_stem
        self.bn1 = model.bn1
        self.act1 = model.act1

        self.block0 = torch.nn.Sequential(*model.blocks[0:layers[0]])
        self.block1 = torch.nn.Sequential(*model.blocks[layers[0]:layers[1]])
        self.block2 = torch.nn.Sequential(*model.blocks[layers[1]:layers[2]])
        self.block3 = torch.nn.Sequential(*model.blocks[layers[2]:layers[3]])
        self.block4 = torch.nn.Sequential(*model.blocks[layers[3]:layers[4]])

        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv_stem(x)))
        x2 = self.block0(x)
        x4 = self.block1(x2)
        # return x4,x4,x4,x4
        x8 = self.block2(x4)
        x16 = self.block3(x8)
        x32 = self.block4(x16)
        return [x4, x8, x16, x32]

class FeatUp(SubModule):
    def __init__(self):
        super(FeatUp, self).__init__()
        chans = [16, 24, 32, 96, 160]
        self.deconv32_16 = Conv2x(chans[4], chans[3], deconv=True, concat=True)
        self.deconv16_8 = Conv2x(chans[3]*2, chans[2], deconv=True, concat=True)
        self.deconv8_4 = Conv2x(chans[2]*2, chans[1], deconv=True, concat=True)
        self.conv4 = BasicConv(chans[1]*2, chans[1]*2, kernel_size=3, stride=1, padding=1)

        self.weight_init()

    def forward(self, featL, featR=None):
        x4, x8, x16, x32 = featL

        y4, y8, y16, y32 = featR
        x16 = self.deconv32_16(x32, x16)
        y16 = self.deconv32_16(y32, y16)
        
        x8 = self.deconv16_8(x16, x8)
        y8 = self.deconv16_8(y16, y8)
        x4 = self.deconv8_4(x8, x4)
        y4 = self.deconv8_4(y8, y4)
        x4 = self.conv4(x4)
        y4 = self.conv4(y4)

        return [x4, x8, x16, x32], [y4, y8, y16, y32]

class DFTM(SubModule):
    r"""
    Dense Fourier Transform Module
    """
    def __init__(self, cv_chan, im_chan):
        super(DFTM, self).__init__()

        self.semantic = nn.Sequential(
            BasicConv(im_chan, im_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(im_chan//2, cv_chan, 1))

        self.FFT2 = nn.Sequential(Three_FT2Block(cv_chan),
                                 nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))
        self.FFT3 = nn.Sequential(Three_FT3Block(cv_chan),
                                 nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))
        self.FFT4 = nn.Sequential(Three_FT4Block(cv_chan),
                                 nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))
        self.FFT5 = nn.Sequential(Three_FT5Block(cv_chan),
                                 nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))

        self.att = nn.Sequential(BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,3,3),
                                 padding=(0,1,1), stride=1, dilation=1),
                                 nn.Conv3d(cv_chan, cv_chan, kernel_size=1, stride=1, padding=0, bias=False))

        self.agg = BasicConv(cv_chan, cv_chan, is_3d=True, bn=True, relu=True, kernel_size=(1,3,3),
                             padding=(0,1,1), stride=1, dilation=1)

        self.weight_init()

    def forward(self, cv, feat):
        '''
        '''
        feat = self.semantic(feat).unsqueeze(2)
        att = feat+cv
        att2 = self.FFT2(att)
        att3 = self.FFT3(att2+att)
        att4 = self.FFT4(att3+att2+att)
        att5 = self.FFT5(att4+att3+att2+att)
        atta = att + att2 + att3 + att4 + att5
        atta = self.att(atta)

        cv = torch.sigmoid(atta)*feat + cv
        cv = self.agg(cv)
        return cv

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

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

class SDI(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.convs = nn.ModuleList(
            [nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1)] * 4)

    def forward(self, xs, anchor):
        ans = torch.ones_like(anchor)
        target_size0 = anchor.shape[-1]
        target_size1 = anchor.shape[-2]

        for i, x in enumerate(xs):
            if x.shape[-1] > target_size0:
                x = F.adaptive_avg_pool2d(x, (target_size1, target_size0))
            elif x.shape[-1] < target_size0:
                x = F.interpolate(x, size=(target_size1, target_size0),
                                      mode='bilinear', align_corners=True)

            ans = ans * self.convs[i](x)

        return ans

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class hourglass_fusion(nn.Module):
    def __init__(self, in_channels):
        super(hourglass_fusion, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 1, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.ca_1 = ChannelAttention(96)
        self.sa_1 = SpatialAttention()

        self.ca_2 = ChannelAttention(64)
        self.sa_2 = SpatialAttention()

        self.ca_3 = ChannelAttention(192)
        self.sa_3 = SpatialAttention()

        self.ca_4 = ChannelAttention(160)
        self.sa_4 = SpatialAttention()


        self.Translayer_1 = BasicConv2d(96, 64, 1)
        self.Translayer_2 = BasicConv2d(64, 64, 1)
        self.Translayer_3 = BasicConv2d(192, 64, 1)
        self.Translayer_4 = BasicConv2d(160, 64, 1)

        self.sdi_1 = SDI(64)
        self.sdi_2 = SDI(64)
        self.sdi_3 = SDI(64)
        self.sdi_4 = SDI(64)

        self.Translayer_1o = BasicConv2d(64, 96, 1)
        self.Translayer_2o = BasicConv2d(64, 64, 1)
        self.Translayer_3o = BasicConv2d(64, 192, 1)
        self.Translayer_4o = BasicConv2d(64, 160, 1)

        self.DFTM_32 = DFTM(in_channels*6, 160)
        self.DFTM_16 = DFTM(in_channels*4, 192)
        self.DFTM_8 = DFTM(in_channels*2, 64)

    def forward(self, x, imgs):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        #Full Multi-scale Fusion Module (FMFM)
        f1 = self.ca_1(imgs[0]) * imgs[0]
        f1 = self.sa_1(f1) * f1
        f1 = self.Translayer_1(f1)

        f2 = self.ca_2(imgs[1]) * imgs[1]
        f2 = self.sa_2(f2) * f2
        f2 = self.Translayer_2(f2)

        f3 = self.ca_3(imgs[2]) * imgs[2]
        f3 = self.sa_3(f3) * f3
        f3 = self.Translayer_3(f3)

        f4 = self.ca_4(imgs[3]) * imgs[3]
        f4 = self.sa_4(f4) * f4
        f4 = self.Translayer_4(f4)

        f41 = self.sdi_4([f1, f2, f3, f4], f4)
        f31 = self.sdi_3([f1, f2, f3, f4], f3)
        f21 = self.sdi_2([f1, f2, f3, f4], f2)

        f41 = self.Translayer_4o(f41)
        f31 = self.Translayer_3o(f31)
        f21 = self.Translayer_2o(f21)

        #DFTM
        conv3 = self.DFTM_32(conv3, f41)
        conv3_up = self.conv3_up(conv3)

        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)

        conv2 = self.DFTM_16(conv2, f31)
        conv2_up = self.conv2_up(conv2)

        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)

        conv1 = self.DFTM_8(conv1, f21)
        conv = self.conv1_up(conv1)

        return conv


class FT_FMF(nn.Module):
    r"""
    FT-FMF Net
    """
    def __init__(self, maxdisp):
        super(FT_FMF, self).__init__()
        self.maxdisp = maxdisp 
        self.feature = Feature()
        self.feature_up = FeatUp()
        chans = [16, 24, 32, 96, 160]

        self.stem_2 = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.BatchNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x(32, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU()
            )

        self.conv = BasicConv(96, 48, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(48, 48, kernel_size=1, padding=0, stride=1)
        self.semantic = nn.Sequential(
            BasicConv(96, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 8, kernel_size=1, padding=0, stride=1, bias=False))
        self.agg = BasicConv(8, 8, is_3d=True, kernel_size=(1,5,5), padding=(0,2,2), stride=1)
        self.hourglass_fusion = hourglass_fusion(8)
        self.corr_stem = BasicConv(1, 8, is_3d=True, kernel_size=3, stride=1, padding=1)

    def forward(self, left, right):
        features_left = self.feature(left)
        features_right = self.feature(right)
        features_left, features_right = self.feature_up(features_left, features_right)
        stem_2x = self.stem_2(left)
        stem_4x = self.stem_4(stem_2x)
        stem_2y = self.stem_2(right)
        stem_4y = self.stem_4(stem_2y)

        features_left[0] = torch.cat((features_left[0], stem_4x), 1)
        features_right[0] = torch.cat((features_right[0], stem_4y), 1)


        match_left = self.desc(self.conv(features_left[0]))
        match_right = self.desc(self.conv(features_right[0]))

        corr_volume = build_norm_correlation_volume(match_left, match_right, self.maxdisp//4)
        corr_volume = self.corr_stem(corr_volume)
        feat_volume = self.semantic(features_left[0]).unsqueeze(2)
        volume = self.agg(feat_volume * corr_volume)
        cost = self.hourglass_fusion(volume, features_left)

        xspx = self.spx_4(features_left[0])
        xspx = self.spx_2(xspx, stem_2x)
        spx_pred = self.spx(xspx)
        spx_pred = F.softmax(spx_pred, 1)

        disp_samples = torch.arange(0, self.maxdisp//4, dtype=cost.dtype, device=cost.device)
        disp_samples = disp_samples.view(1, self.maxdisp//4, 1, 1).repeat(cost.shape[0],1,cost.shape[3],cost.shape[4])
        pred = regression_topk(cost.squeeze(1), disp_samples, 2)
        pred_up = context_upsample(pred, spx_pred)


        if self.training:
            return [pred_up*4, pred.squeeze(1)*4]

        else:
            return [pred_up*4]

