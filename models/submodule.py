from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np

from functools import partial
from timm.models.layers import trunc_normal_, DropPath
import os
import sys
import torch.fft
import math

import traceback



class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size, padding=padding,
                               stride=stride, dilation=dilation, groups=dim_in)
        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


if 'DWCONV_IMPL' in os.environ:
    try:
        sys.path.append(os.environ['DWCONV_IMPL'])
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM


        def get_dwconv(dim, kernel, bias):
            return DepthWiseConv2dImplicitGEMM(dim, kernel, bias)
        # print('Using Megvii large kernel dw conv impl')
    except:
        print(traceback.format_exc())


        def get_dwconv(dim, kernel, bias):
            return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

        # print('[fail to use Megvii Large kernel] Using PyTorch large kernel dw conv impl')
else:
    def get_dwconv(dim, kernel, bias):
        return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

    # print('Using PyTorch large kernel dw conv impl')


class gnconv3D(nn.Module):
    r"""
    3FT-gnconv
    """
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv3d(dim, 2 * dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv3d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv3d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s

        print('[gnconv3D]', order, 'order with dims=', self.dims, 'scale=%.4f' % self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W, L = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x)

        return x


class Block(nn.Module):
    r"""
    3FTnBlock
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv3D):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                   requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W, L = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 4, 1, 2, 3)

        x = input + self.drop_path(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class ThreeFFT(nn.Module):
    r"""
    3FFT
    """
    def __init__(self, dim, h=14, w=4,l=4):
        super().__init__()
        self.dw = nn.Conv3d(dim, dim, bias=False,kernel_size=(1,3,3),padding=(0,1,1), stride=1, dilation=1)
        self.complex_weight = nn.Parameter(torch.randn(dim, h, w, l, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1 = self.dw(x)

        x2 = x.to(torch.float32)
        B, C, a, b, c = x2.shape
        x2 = torch.fft.rfftn(x, dim=(2, 3, 4), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:4] == x2.shape[2:5]:
            weight = F.interpolate(weight.permute(4, 0, 1, 2, 3),
                                   size=x2.shape[2:5],
                                   mode='trilinear',
                                   align_corners=True).permute(1, 2, 3, 4, 0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfftn(x2, s=(a, b, c), dim=(2, 3, 4), norm='ortho')

        x = x1+x2

        x = self.post_norm(x)
        return x

class Three_FT2Block(nn.Module):
    r"""
    3FF2Block
    """
    def __init__(self, c1, gnconv=gnconv3D, block=Block):
        super(Three_FT2Block, self).__init__()

        if not isinstance(gnconv, list):
            gnconv = [partial(gnconv, order=2, s=1 / 3, gflayer=ThreeFFT),
                      partial(gnconv, order=3, s=1 / 3, gflayer=ThreeFFT),
                      partial(gnconv, order=4, s=1 / 3, h=24, w=13, gflayer=ThreeFFT),
                      partial(gnconv, order=5, s=1 / 3, h=12, w=7, gflayer=ThreeFFT)]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        if isinstance(gnconv[0], str):
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)

        self.gnconv = nn.Sequential(
            *[block(dim=c1, drop_path=0.3,
                    layer_scale_init_value=1e-6, gnconv=gnconv[0]) for j in range(1)],
        )
    def forward(self, x):
        return self.gnconv(x)

    def forward_fuse(self, x):
        return self.gnconv(x)

class Three_FT3Block(nn.Module):
    r"""
    3FF3Block
    """
    def __init__(self, c1, gnconv=gnconv3D, block=Block):
        super(Three_FT3Block, self).__init__()

        if not isinstance(gnconv, list):
            gnconv = [partial(gnconv, order=2, s=1 / 3, gflayer=ThreeFFT),
                      partial(gnconv, order=3, s=1 / 3, gflayer=ThreeFFT),
                      partial(gnconv, order=4, s=1 / 3, h=24, w=13, gflayer=ThreeFFT),
                      partial(gnconv, order=5, s=1 / 3, h=12, w=7, gflayer=ThreeFFT)]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        if isinstance(gnconv[0], str):
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)

        self.gnconv = nn.Sequential(
            *[block(dim=c1, drop_path=0.3,
                    layer_scale_init_value=1e-6, gnconv=gnconv[1]) for j in range(1)],
        )
    def forward(self, x):
        return self.gnconv(x)

    def forward_fuse(self, x):
        return self.gnconv(x)

class Three_FT4Block(nn.Module):
    r"""
    3FF4Block
    """
    def __init__(self, c1, gnconv=gnconv3D, block=Block):
        super(Three_FT4Block, self).__init__()

        if not isinstance(gnconv, list):
            gnconv = [partial(gnconv, order=2, s=1 / 3, gflayer=ThreeFFT),
                      partial(gnconv, order=3, s=1 / 3, gflayer=ThreeFFT),
                      partial(gnconv, order=4, s=1 / 3, h=24, w=13, gflayer=ThreeFFT),
                      partial(gnconv, order=5, s=1 / 3, h=12, w=7, gflayer=ThreeFFT)]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        if isinstance(gnconv[0], str):
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)

        self.gnconv = nn.Sequential(
            *[block(dim=c1, drop_path=0.3,
                    layer_scale_init_value=1e-6, gnconv=gnconv[2]) for j in range(1)],
        )
    def forward(self, x):
        return self.gnconv(x)

    def forward_fuse(self, x):
        return self.gnconv(x)

class Three_FT5Block(nn.Module):
    r"""
    3FF5Block
    """
    def __init__(self, c1, gnconv=gnconv3D, block=Block):
        super(Three_FT5Block, self).__init__()

        if not isinstance(gnconv, list):
            gnconv = [partial(gnconv, order=2, s=1 / 3, gflayer=ThreeFFT),
                      partial(gnconv, order=3, s=1 / 3, gflayer=ThreeFFT),
                      partial(gnconv, order=4, s=1 / 3, h=24, w=13, gflayer=ThreeFFT),
                      partial(gnconv, order=5, s=1 / 3, h=12, w=7, gflayer=ThreeFFT)]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        if isinstance(gnconv[0], str):
            print('[GConvNet]: convert str gconv to func')
            gnconv = [eval(g) for g in gnconv]

        if isinstance(block, str):
            block = eval(block)

        self.gnconv = nn.Sequential(
            *[block(dim=c1, drop_path=0.3,
                    layer_scale_init_value=1e-6, gnconv=gnconv[3]) for j in range(1)],
        )
    def forward(self, x):
        return self.gnconv(x)

    def forward_fuse(self, x):
        return self.gnconv(x)

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d 
        if deconv and is_3d: 
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat: 
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(
                x,
                size=(rem.shape[-2], rem.shape[-1]),
                mode='nearest')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else: 
            x = x + rem
        x = self.conv2(x)
        return x


def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)

def groupwise_difference(fea1, fea2, num_groups):
    B, G, C, H, W = fea1.shape
    cost = torch.pow((fea1 - fea2), 2).sum(2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_struct_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, G, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_difference(refimg_fea[:, :, :, :, i:], targetimg_fea[:, :, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_difference(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, :] = refimg_fea[:, :, :, :]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume


def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def groupwise_correlation_norm(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1 = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2 = fea2.view([B, num_groups, channels_per_group, H, W])
    cost = ((fea1/(torch.norm(fea1, 2, 2, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 2, True)+1e-05))).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost


def build_gwc_volume_norm(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation_norm(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation_norm(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume


def norm_correlation(fea1, fea2):
    cost = torch.mean(((fea1/(torch.norm(fea1, 2, 1, True)+1e-05)) * (fea2/(torch.norm(fea2, 2, 1, True)+1e-05))), dim=1, keepdim=True)
    return cost

def build_norm_correlation_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 1, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = norm_correlation(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
        else:
            volume[:, :, i, :, :] = norm_correlation(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

def SpatialTransformer_grid(x, y, disp_range_samples):

    bs, channels, height, width = y.size()
    ndisp = disp_range_samples.size()[1]

    mh, mw = torch.meshgrid([torch.arange(0, height, dtype=x.dtype, device=x.device),
                                 torch.arange(0, width, dtype=x.dtype, device=x.device)])  # (H *W)
    mh = mh.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)
    mw = mw.reshape(1, 1, height, width).repeat(bs, ndisp, 1, 1)  # (B, D, H, W)

    cur_disp_coords_y = mh
    cur_disp_coords_x = mw - disp_range_samples
    coords_x = cur_disp_coords_x / ((width - 1.0) / 2.0) - 1.0  # trans to -1 - 1
    coords_y = cur_disp_coords_y / ((height - 1.0) / 2.0) - 1.0
    grid = torch.stack([coords_x, coords_y], dim=4) #(B, D, H, W, 2)

    y_warped = F.grid_sample(y, grid.view(bs, ndisp * height, width, 2), mode='bilinear',
                               padding_mode='zeros', align_corners=True).view(bs, channels, ndisp, height, width)  #(B, C, D, H, W)

    return y_warped
        
        
        
def context_upsample(depth_low, up_weights):
    ###
    # cv (b,1,h,w)
    # sp (b,9,4*h,4*w)
    ###
    b, c, h, w = depth_low.shape
        
    depth_unfold = F.unfold(depth_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    depth_unfold = F.interpolate(depth_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    depth = (depth_unfold*up_weights).sum(1)
        
    return depth


def regression_topk(cost, disparity_samples, k):

    _, ind = cost.sort(1, True)
    pool_ind = ind[:, :k]
    cost = torch.gather(cost, 1, pool_ind)
    prob = F.softmax(cost, 1)
    disparity_samples = torch.gather(disparity_samples, 1, pool_ind)    
    pred = torch.sum(disparity_samples * prob, dim=1, keepdim=True)
    return pred


