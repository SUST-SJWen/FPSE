import copy
import math
import logging
import time

import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
from timm.models.swin_transformer import BasicLayer, PatchMerging
from timm.models.layers import trunc_normal_
from torchvision.transforms import transforms
from utils import *
import numpy as np
from skimage import io
from einops import rearrange
from einops.layers.torch import Rearrange
from torch.nn.modules.utils import _pair
from os.path import join as pjoin
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from .MSDConv_SSFC import MSDConv_SSFC
from .DeformConv import DeformConv2d
import scipy.misc

from .swin_transformer_unet_skip_expand_decoder_sys import PatchEmbed


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
logger = logging.getLogger(__name__)
device = torch.device("cuda:0" )

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

class Attention(nn.Module):
    def __init__(self, dim, factor, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim * factor),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SwinTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, embed_dim=96,
                 depths=[2, 2, 6], num_heads=[3, 6, 12],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, **kwargs):

        super().__init__()

        patches_resolution = [img_size // patch_size, img_size // patch_size]
        num_patches = patches_resolution[0] * patches_resolution[1]

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               out_dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None)
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


class SpatialAttention(nn.Module):
    def __init__(self, in_channel=512):
        super(SpatialAttention, self).__init__()

        # assert  kernel_size in (3,7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.conv1 = nn.Conv2d(in_channel, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out_c = torch.mean(x, dim=1, keepdim=True)
        avg_out_s = self.avgpool(x)
        avg_out_s = avg_out_s.repeat(1, 1, x.size()[2], x.size()[3])
        out = avg_out_s + x
        out = self.conv1(out)
        weight=self.sigmoid(out)
        out=weight*x+x
        return out


class PyramidFeatures(nn.Module):
    def __init__(self, config, img_size=224, in_channels=3):
        super().__init__()

        model_path = config.swin_pretrained_path
        self.swin_transformer = SwinTransformer(img_size, in_chans=in_channels)
        checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
        unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight",
                      "patch_embed.norm.bias",
                      "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
                      "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight",
                      "layers.1.downsample.norm.bias",
                      "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight",
                      "layers.2.downsample.norm.bias",
                      "layers.2.downsample.reduction.weight", "norm.weight", "norm.bias"]

        resnet = eval(f"torchvision.models.{config.cnn_backbone}(pretrained={config.resnet_pretrained})")
        self.resnet_layers = nn.ModuleList(resnet.children())[:7]
        self.resnet_layers[0] = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                          bias=False)
        # self.sa=SpatialAttention()

        self.p1_ch = nn.Conv2d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0], kernel_size=1)
        self.p1_pm = PatchMerging((config.image_size // config.patch_size, config.image_size // config.patch_size),
                                  config.swin_pyramid_fm[0])
        self.p1_pm.state_dict()['reduction.weight'][:] = checkpoint["layers.0.downsample.reduction.weight"]
        self.p1_pm.state_dict()['norm.weight'][:] = checkpoint["layers.0.downsample.norm.weight"]
        self.p1_pm.state_dict()['norm.bias'][:] = checkpoint["layers.0.downsample.norm.bias"]
        self.norm_1 = nn.LayerNorm(config.swin_pyramid_fm[0])
        self.avgpool_1 = nn.AdaptiveAvgPool1d(1)

        self.p2 = self.resnet_layers[5]
        self.p2_ch = nn.Conv2d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1], kernel_size=1)
        self.p2_pm = PatchMerging(
            (config.image_size // config.patch_size // 2, config.image_size // config.patch_size // 2),
            config.swin_pyramid_fm[1])
        self.p2_pm.state_dict()['reduction.weight'][:] = checkpoint["layers.1.downsample.reduction.weight"]
        self.p2_pm.state_dict()['norm.weight'][:] = checkpoint["layers.1.downsample.norm.weight"]
        self.p2_pm.state_dict()['norm.bias'][:] = checkpoint["layers.1.downsample.norm.bias"]
        self.norm_2 = nn.LayerNorm(config.swin_pyramid_fm[1])
        self.avgpool_2 = nn.AdaptiveAvgPool1d(1)

        self.p3 = self.resnet_layers[6]
        self.p3_ch = nn.Conv2d(config.cnn_pyramid_fm[2], config.swin_pyramid_fm[2], kernel_size=1)
        self.norm_3 = nn.LayerNorm(config.swin_pyramid_fm[2])
        self.avgpool_3 = nn.AdaptiveAvgPool1d(1)

        for key in list(checkpoint.keys()):
            if key in unexpected or 'layers.3' in key:
                del checkpoint[key]
        self.swin_transformer.load_state_dict(checkpoint)

    def forward(self, x):
        for i in range(5):
            x = self.resnet_layers[i](x)

            # Level 1
        fm1 = x
        fm1_ch = self.p1_ch(x)
        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(fm1_ch)
        sw1 = self.swin_transformer.layers[0](fm1_reshaped)
        sw1_skipped = fm1_reshaped + sw1
        # sw1_skipped1=sw1_skipped+self.sa(fm1).flatten(2).transpose(1, 2)*sw1_skipped
        norm1 = self.norm_1(sw1_skipped)
        sw1_CLS = self.avgpool_1(norm1.transpose(1, 2))
        sw1_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(sw1_CLS)
        fm1_sw1 = self.p1_pm(sw1)

        # Level 2
        fm1_sw2 = self.swin_transformer.layers[1](fm1_sw1)
        fm2 = self.p2(fm1)
        fm2_ch = self.p2_ch(fm2)
        fm2_reshaped = Rearrange('b c h w -> b (h w) c')(fm2_ch)
        fm2_sw2_skipped = fm2_reshaped + fm1_sw2
        # fm2_sw2_skipped1=fm2_sw2_skipped+self.sa(fm2).flatten(2).transpose(1, 2)*fm2_sw2_skipped
        norm2 = self.norm_2(fm2_sw2_skipped)
        sw2_CLS = self.avgpool_2(norm2.transpose(1, 2))
        sw2_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(sw2_CLS)
        fm2_sw2 = self.p2_pm(fm1_sw2)

        # Level 3
        fm2_sw3 = self.swin_transformer.layers[2](fm2_sw2)
        fm3 = self.p3(fm2)
        fm3_ch = self.p3_ch(fm3)
        fm3_reshaped = Rearrange('b c h w -> b (h w) c')(fm3_ch)
        fm3_sw3_skipped = fm3_reshaped + fm2_sw3
        # fm3_sw3_skipped1=fm3_sw3_skipped+self.sa(fm3).flatten(2).transpose(1, 2)*fm3_sw3_skipped
        norm3 = self.norm_3(fm3_sw3_skipped)
        sw3_CLS = self.avgpool_3(norm3.transpose(1, 2))
        sw3_CLS_reshaped = Rearrange('b c 1 -> b 1 c')(sw3_CLS)

        return [fm1, fm2, fm3, fm3_reshaped, fm2_sw3, fm3_sw3_skipped]

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,kernel_size=3,padding=1):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size,stride=1,padding=padding,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size,stride=stride,padding=padding,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class DeformConv_block(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,kernel_size=3,dilation=1,padding=1,DE="Down"):
        super(DeformConv_block,self).__init__()
        self.conv_Deform=nn.Sequential(
            DeformConv2d(ch_in, int(ch_out/2), kernel_size=kernel_size, stride=stride,dilation=dilation, padding=dilation, bias=True,device="cuda:0",DE=DE),
            nn.BatchNorm2d(int(ch_out/2)),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, int(ch_out/2), kernel_size=kernel_size, stride=stride, dilation=dilation, padding=dilation, bias=True),
            nn.BatchNorm2d(int(ch_out/2)),
            nn.ReLU(inplace=True)
        )
        self.conv1x1=nn.Conv2d(ch_out,ch_out,1)

    def forward(self,x):
        x1 = self.conv_Deform(x)
        x2=self.conv(x)
        out=self.conv1x1(torch.cat((x1,x2),1))
        return out

class My_Conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,kernel_size=3,dilation=1,padding=1,DE="Down"):
        super(My_Conv_block,self).__init__()
        self.stride=stride
        self.Conv1x1=nn.Conv2d(ch_in,int(ch_out/2),1)
        self.conv_Deform=nn.Sequential(
            DeformConv2d(int(ch_out/2), int(ch_out/2), kernel_size=kernel_size, stride=stride,dilation=dilation, padding=dilation, bias=True,device="cuda:0",DE=DE),
            nn.BatchNorm2d(int(ch_out/2)),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(int(ch_out/2), int(ch_out/2), kernel_size=kernel_size, stride=stride, dilation=dilation, padding=dilation, bias=True),
            nn.BatchNorm2d(int(ch_out/2)),
            nn.ReLU(inplace=True)
        )
        self.conv1x1=nn.Conv2d(ch_out,ch_out,1)
        self.pool=nn.MaxPool2d(2)

    def forward(self,x):
        x_c=self.Conv1x1(x)
        x1 = self.conv_Deform(x_c)
        x2=self.conv(x_c)
        if self.stride==1:
            out = self.conv1x1(torch.cat((x1, x2+x_c), 1))
        else:
            out=self.conv1x1(torch.cat((x1,x2+self.pool(x_c)),1))
        return out

class My_Deconv_block(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,kernel_size=3,dilation=1,padding=1,DE="UP"):
        super(My_Deconv_block,self).__init__()
        self.Conv1x1=nn.Conv2d(ch_in,int(ch_out/2),1)
        self.conv_Deform = nn.Sequential(
            DeformConv2d(int(ch_out/2), int(ch_out / 2), kernel_size=kernel_size, stride=stride, dilation=dilation,
                         padding=dilation, bias=True, device="cuda:0", DE=DE),
            nn.BatchNorm2d(int(ch_out / 2)),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(int(ch_out/2), int(ch_out / 2), kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=dilation, output_padding=stride - 1, bias=True),
            nn.BatchNorm2d(int(ch_out / 2)),
            nn.ReLU(inplace=True)
        )
        self.conv1x1=nn.Conv2d(ch_out,ch_out,1)
        self.up=nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

    def forward(self,x):
        x_c=self.Conv1x1(x)
        x1 = self.conv_Deform(x_c)
        x2=self.conv(x_c)
        out=self.conv1x1(torch.cat((x1,x2+self.up(x_c)),1))
        return out
from functools import partial
class Channel_Attention(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(Channel_Attention, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=4,stride=4)
        self.avg_pool = nn.AvgPool2d(kernel_size=4,stride=4)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel // reduction, out_features=in_channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 通道注意力机制
        h,w=x.size(2),x.size(3)
        avgout = self.mlp(x.flatten(2).transpose(1, 2))
        # avgout = Rearrange('b c (h w) -> b c h w', h=h, w=w)(avgout)
        channel_out = self.sigmoid(avgout).transpose(1, 2)
        channel_out = Rearrange('b c (h w) -> b c h w', h=h, w=w)(channel_out)
        out_weight = avgout.transpose(1, 2)
        out_weight = Rearrange('b c (h w) -> b c h w', h=h, w=w)(out_weight)
        # channel_out = torch.nn.UpsamplingNearest2d(scale_factor=4)(channel_out)
        # channel_out = channel_out.view(x.size(0), x.size(1), 1, 1)
        channel_out = channel_out * x
        return channel_out,out_weight

class Spatial_Attention(nn.Module):
    def __init__(self,kernel_size=7):
        super(Spatial_Attention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat((max_out, mean_out), dim=1)
        out_weight = self.conv(out)
        out = self.sigmoid(out_weight)
        out = out * x+x
        return out
def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    return torch.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    '''
    filter_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3],
    ]).astype(np.float32)
    filter_y = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3],
    ]).astype(np.float32)
    '''
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y
# class My_Conv_block1(nn.Module):
#     def __init__(self,ch_in,ch_out,stride=1,kernel_size=3,dilation=1,padding=1,DE="Down"):
#         super(My_Conv_block1,self).__init__()
#         self.stride=stride
#         self.Conv1x1=nn.Conv2d(ch_in,int(ch_out/2),1)
#         self.conv_Deform=nn.Sequential(
#             DeformConv2d(int(ch_out/2), int(ch_out/2), kernel_size=kernel_size, stride=1,dilation=dilation, padding=dilation, bias=True,device="cuda:0",DE=DE),
#             nn.BatchNorm2d(int(ch_out/2)),
#             nn.ReLU(inplace=True)
#         )
#         self.conv = nn.Sequential(
#             nn.Conv2d(int(ch_out/2), int(ch_out/2), kernel_size=kernel_size, stride=stride, dilation=dilation, padding=dilation, bias=True),
#             nn.BatchNorm2d(int(ch_out/2)),
#             nn.ReLU(inplace=True)
#         )
#         self.sobel_x, self.sobel_y = get_sobel(int(ch_out/2), 1)
#         self.sa = Spatial_Attention()
#         self.conv1x1=nn.Conv2d(ch_out,ch_out,1)
#
#     def forward(self,x):
#         x_c=self.Conv1x1(x)
#         x_conv=self.conv(x_c)
#         x_L_sa, out_weight_sa = self.sa(x_conv)
#         s = run_sobel(self.sobel_x, self.sobel_y, x_conv)
#         out = self.conv_Deform(x_L_sa+s)
#         out=self.conv1x1(torch.cat((x_conv,out),dim=1))
#         return out
class My_Conv_block1(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,kernel_size=3,dilation=1,padding=1,DE="Down"):
        super(My_Conv_block1,self).__init__()
        self.stride=stride
        self.conv1x1=nn.Conv2d(ch_in,int(ch_out/2),kernel_size=1)
        self.maxpool=nn.MaxPool2d(stride)
        self.conv_Deform1=nn.Sequential(
            DeformConv2d(int(ch_out/2), int(ch_out/2), kernel_size=3, stride=1,dilation=1, padding=1, bias=True,device="cuda:0",DE=DE),
            nn.BatchNorm2d(int(ch_out/2)),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(int(ch_out/2), int(ch_out/2), kernel_size=kernel_size, stride=1, dilation=dilation, padding=dilation, bias=True),
            nn.BatchNorm2d(int(ch_out/2)),
            nn.ReLU(inplace=True)
        )
        self.sobel_x, self.sobel_y = get_sobel(int(ch_out/2), 1)
        self.sa = SpatialAttention(int(ch_out/2))
        self.Conv1x1=nn.Conv2d(ch_out,ch_out,kernel_size=1)

    def forward(self,x):
        x=self.maxpool(x)
        x=self.conv1x1(x)
        x_L_sa = self.sa(x)
        out1 = self.conv_Deform1(x_L_sa)
        out1 = self.conv1(out1)
        output=self.Conv1x1(torch.cat((x_L_sa,out1),dim=1))
        return output
class My_Deconv_block1(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,kernel_size=3,dilation=1,padding=1,DE="UP"):
        super(My_Deconv_block1,self).__init__()
        self.Conv1x1=nn.Conv2d(ch_in,int(ch_out/2),1)
        self.conv_Deform = nn.Sequential(
            DeformConv2d(int(ch_out/2), int(ch_out / 2), kernel_size=kernel_size, stride=stride, dilation=dilation,
                         padding=dilation, bias=True, device="cuda:0", DE=DE),
            nn.BatchNorm2d(int(ch_out / 2)),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(int(ch_out/2), int(ch_out / 2), kernel_size=kernel_size, stride=1, dilation=dilation,
                               padding=dilation, output_padding=0, bias=True),
            nn.BatchNorm2d(int(ch_out / 2)),
            nn.ReLU(inplace=True)
        )
        self.conv1x1=nn.Conv2d(ch_out,ch_out,1)
        self.up=nn.Upsample(scale_factor=2, mode='bilinear',align_corners=True)

    def forward(self,x):
        x_c=self.Conv1x1(x)
        x1 = self.conv_Deform(x_c)
        x2=self.conv(self.up(x_c)+x1)
        out=self.conv1x1(torch.cat((x2,self.up(x_c)),1))
        return out
class MSDconv_block(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,kernel_size=3,padding=1):
        super(MSDconv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(2),
            MSDConv_SSFC(ch_in, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            MSDConv_SSFC(ch_out, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class conv_block_decoder(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(conv_block_decoder,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3,
                               stride=stride, padding=1, output_padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Deformconv_block_decoder(nn.Module):
    def __init__(self,ch_in,ch_out,stride=2,kernel_size=3,dilation=1,padding=1,DE="UP"):
        super(Deformconv_block_decoder,self).__init__()
        self.conv_Deform=nn.Sequential(
            DeformConv2d(ch_in, int(ch_out/2), kernel_size=kernel_size, stride=stride,dilation=dilation, padding=dilation, bias=True,device="cuda:0",DE=DE),
            nn.BatchNorm2d(int(ch_out/2)),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(ch_in, int(ch_out/2), kernel_size=kernel_size, stride=stride, dilation=dilation, padding=dilation, output_padding=stride-1, bias=True),
            nn.BatchNorm2d(int(ch_out/2)),
            nn.ReLU(inplace=True)
        )
        self.conv1x1=nn.Conv2d(ch_out,ch_out,1)

    def forward(self,x):
        x1 = self.conv_Deform(x)
        x2=self.conv(x)
        out=self.conv1x1(torch.cat((x1,x2),1))
        return out
class MSDconv_block_decoder(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(MSDconv_block_decoder,self).__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            MSDConv_SSFC(ch_in, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            MSDConv_SSFC(ch_out, ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x


from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

savepath = r'../features'
if not os.path.exists(savepath):
    os.mkdir(savepath)
def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(16, 16))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        print("{}/{}".format(i, width * height))
    fig.savefig(savename, dpi=100)
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))
class bottleneck(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,r=4):
        super(bottleneck,self).__init__()
        self.maxpool=nn.MaxPool2d(stride)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//r, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_out//r),
            nn.ReLU(inplace=True),
        )
        self.conv2=nn.Sequential(
             nn.Conv2d(ch_out//r, ch_out // r, kernel_size=3, stride=1, padding=1, bias=True),
             nn.BatchNorm2d(ch_out//r),
             nn.ReLU(inplace=True),
        )
        self.conv3=nn.Sequential(
             nn.Conv2d(ch_out // r, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
             nn.BatchNorm2d(ch_out),
             nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
class inception_module(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,r=4):
        super(inception_module,self).__init__()
        self.maxpool=nn.MaxPool2d(stride)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//r, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_out//r),
            nn.ReLU(inplace=True),
        )
        self.conv2_1=nn.Sequential(
             nn.Conv2d(ch_in, ch_out // r, kernel_size=1, stride=1, padding=0, bias=True),
             nn.BatchNorm2d(ch_out//r),
             nn.ReLU(inplace=True),
        )
        self.conv2_2 = nn.Sequential(
            nn.Conv2d( ch_out // r, ch_out // r, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out // r),
            nn.ReLU(inplace=True),
        )
        self.conv3_1=nn.Sequential(
             nn.Conv2d(ch_in, ch_out// r, kernel_size=1, stride=1, padding=0, bias=True),
             nn.BatchNorm2d(ch_out// r),
             nn.ReLU(inplace=True),
        )
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(ch_out// r, ch_out // r, kernel_size=5, stride=1, padding=2, bias=True),
            nn.BatchNorm2d(ch_out // r),
            nn.ReLU(inplace=True),
        )
        self.maxpool4=nn.MaxPool2d(kernel_size=3,padding=1,stride=1)
        self.conv4_1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // r, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out // r),
            nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x=self.maxpool(x)
        a = self.conv1(x)
        b=self.conv2_1(x)
        b=self.conv2_2(b)
        c = self.conv3_1(x)
        c = self.conv3_2(c)
        d = self.maxpool4(x)
        d=self.conv4_1(d)
        return torch.cat((a,b,c,d),dim=1)
class residual_block(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,r=4):
        super(residual_block,self).__init__()
        self.maxpool=nn.MaxPool2d(stride)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//r, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_out//r),
            nn.ReLU(inplace=True),
        )
        self.conv2=nn.Sequential(
             nn.Conv2d(ch_out//r, ch_out // r, kernel_size=3, stride=1, padding=1, bias=True),
             nn.BatchNorm2d(ch_out//r),
             nn.ReLU(inplace=True),
        )
        self.conv3=nn.Sequential(
             nn.Conv2d(ch_out // r, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
             nn.BatchNorm2d(ch_out),
             nn.ReLU(inplace=True),
        )

    def forward(self,x):
        m = self.maxpool(x)
        m = self.conv1(m)
        m = self.conv2(m)
        m = self.conv3(m)
        return x+m
class se_block(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1,r=4):
        super(se_block,self).__init__()
        self.maxpool=nn.MaxPool2d(stride)
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out//r, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(ch_out//r),
            nn.ReLU(inplace=True),
        )
        self.conv2=nn.Sequential(
             nn.Conv2d(ch_out//r, ch_out // r, kernel_size=3, stride=1, padding=1, bias=True),
             nn.BatchNorm2d(ch_out//r),
             nn.ReLU(inplace=True),
        )
        self.conv3=nn.Sequential(
             nn.Conv2d(ch_out // r, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
             nn.BatchNorm2d(ch_out),
             nn.ReLU(inplace=True),
        )

    def forward(self,x):
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
class Encoder_PyramidFeatures1_1(nn.Module):
    def __init__(self, config, img_size=224, in_channels=3,num_classes=9):
        super().__init__()

        model_path = 'weights/swin_tiny_patch4_window7_224.pth'
        self.swin_transformer = SwinTransformer(img_size, in_chans=3)
        checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
        unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight",
                      "patch_embed.norm.bias",
                      "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
                      "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight",
                      "layers.1.downsample.norm.bias",
                      "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight",
                      "layers.2.downsample.norm.bias",
                      "layers.2.downsample.reduction.weight", "norm.weight", "norm.bias"]

        resnet = eval(f"torchvision.models.{config.cnn_backbone}(pretrained={config.resnet_pretrained})")
        self.resnet_layers = nn.ModuleList(resnet.children())[:7]
        self.num_classes=num_classes
        self.p1_sw = nn.Conv2d(256, 96, kernel_size=3,stride=1,padding=1)
        self.p1_ch=nn.Conv2d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0], kernel_size=1)
        # self.p1_ch = My_Conv_block1(256, 96, stride=1, dilation=1)
        # self.p1_ch = conv_block(256, 96)
        # self.block1_1 = Block(vis=True, hidden_size=96, mlp_dim=384, num_attention_heads=3)
        # self.block1_2 = Block(vis=True, hidden_size=96, mlp_dim=384, num_attention_heads=3)
        # self.p1_cat = nn.Conv2d(96 * 2, 96, kernel_size=1)

        self.p2 = self.resnet_layers[5]
        self.p2_ch = nn.Conv2d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1], kernel_size=1)
        self.p2_sw = nn.Conv2d(96, 192, kernel_size=3,stride=2,padding=1)
        # self.p2_ch = My_Conv_block1(96, 192, stride=2, dilation=1)
        # self.p2_ch = conv_block(96, 192,stride=2)
        # self.p2_cat = nn.Conv2d(192 * 2, 192, kernel_size=1)
        # self.block2_1 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=6)
        # self.block2_2 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=6)

        # self.p3 = self.resnet_layers[6]
        self.p3 = self.resnet_layers[6]
        self.p3_ch = nn.Conv2d(config.cnn_pyramid_fm[2],384, kernel_size=1)
        self.p3_sw = nn.Conv2d(192, 384,  kernel_size=3,stride=2,padding=1)
        # self.p3_ch = My_Conv_block1(192, 384, stride=2, dilation=1)
        # self.p3_ch = conv_block(192, 384, stride=2)
        # self.p3_cat = nn.Conv2d(384 * 2, 384, kernel_size=1)
        # self.block3_1 = Block(vis=True, hidden_size=384, mlp_dim=384 * 4, num_attention_heads=12)
        # self.block3_2 = Block(vis=True, hidden_size=384, mlp_dim=384 * 4, num_attention_heads=12)

        for key in list(checkpoint.keys()):
            if key in unexpected or 'layers.3' in key:
                del checkpoint[key]
        self.swin_transformer.load_state_dict(checkpoint)

    def forward(self, x):

        for i in range(5):
            x = self.resnet_layers[i](x)
            # Level 1
        fm1=x
        draw_features(8, 8, x.detach().cpu().numpy(), "{}/x.png".format(savepath))
        fm1_dw = self.p1_ch(fm1)
        B, C, H, W = fm1.shape
        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(fm1)
        fm1_ch = self.p1_sw(x)
        fm1_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm1_ch)
        sw1 = self.swin_transformer.layers[0](fm1_ch_reshaped)
        # sw1_1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw1)
        draw_features(8, 8, fm1_dw.detach().cpu().numpy(), "{}/Skin_CNN1.png".format(savepath))
        # fm1 = self.block1_1(fm1_reshaped, sw1)
        # fm1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(fm1)
        # sw1 = self.block1_2(sw1, fm1_reshaped)
        sw1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw1)
        draw_features(8, 8, sw1.detach().cpu().numpy(), "{}/Skin_Transformer1.png".format(savepath))
        draw_features(8, 8, abs(fm1_dw-sw1).detach().cpu().numpy(), "{}/Skin_cha1.png".format(savepath))

        # Level 2
        fm2_ch = self.p2_sw(sw1)
        fm2_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm2_ch)
        sw2 = self.swin_transformer.layers[1](fm2_ch_reshaped)
        fm2=self.p2(fm1)
        fm2_dw = self.p2_ch(fm2)
        draw_features(8, 8, fm2_dw.detach().cpu().numpy(), "{}/Skin_CNN2.png".format(savepath))
        B, C, H, W = fm2.shape
        # fm2_reshaped = Rearrange('b c h w -> b (h w) c')(fm2)
        # fm2 = self.block2_1(fm2_reshaped, sw2)
        # fm2 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(fm2)
        # sw2 = self.block2_2(sw2, fm2_reshaped)
        sw2 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw2)
        draw_features(8, 8, sw2.detach().cpu().numpy(), "{}/Skin_Transformer2.png".format(savepath))
        draw_features(8, 8, abs(fm2_dw - sw2).detach().cpu().numpy(), "{}/Skin_cha2.png".format(savepath))

        # Level 3
        fm3_ch = self.p3_sw(sw2)
        fm3_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm3_ch)
        sw3 = self.swin_transformer.layers[2](fm3_ch_reshaped)
        fm3=self.p3(fm2)
        fm3_dw = self.p3_ch(fm3)
        draw_features(8, 8, fm3_dw.detach().cpu().numpy(), "{}/Skin_CNN3.png".format(savepath))
        B, C, H, W = fm3.shape
        # fm3_reshaped = Rearrange('b c h w -> b (h w) c')(fm3)
        # fm3 = self.block3_1(fm3_reshaped, sw3)
        # fm3 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(fm3)
        # sw3 = self.block3_2(sw3, fm3_reshaped)
        sw3 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw3)
        draw_features(8, 8, sw3.detach().cpu().numpy(), "{}/Skin_Transformer3.png".format(savepath))
        draw_features(8, 8, abs(fm3_dw - sw3).detach().cpu().numpy(), "{}/Skin_cha3.png".format(savepath))

        return sw1,fm1_dw,sw2,fm2_dw,sw3,fm3_dw
class Dncoder_PyramidFeatures1_1(nn.Module):
    def __init__(self, config, img_size=224, in_channels=3):
        super().__init__()

        model_path = config.swin_pretrained_path
        self.swin_transformer = SwinTransformer(img_size, in_chans=3)
        checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
        unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight",
                      "patch_embed.norm.bias",
                      "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
                      "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight",
                      "layers.1.downsample.norm.bias",
                      "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight",
                      "layers.2.downsample.norm.bias",
                      "layers.2.downsample.reduction.weight", "norm.weight", "norm.bias"]

        # self.die2 = DIE(192,192)
        # self.die1 = DIE(96,96)

        self.p1_sw = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=3,
                                        stride=2, padding=1, output_padding=1)
        self.p1_ch = My_Deconv_block1(192, 96,stride=2,dilation=1)
        self.p1_cat = nn.Conv2d(96 * 2, 96, kernel_size=1)
        self.block1_1 = Block(vis=True, hidden_size=96, mlp_dim=384, num_attention_heads=3)
        self.block1_2 = Block(vis=True, hidden_size=96, mlp_dim=384, num_attention_heads=3)

        self.p2_sw = nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=3,
                                        stride=2, padding=1, output_padding=1)
        self.p2_ch = My_Deconv_block1(384, 192,stride=2,dilation=1)
        self.p2_cat = nn.Conv2d(192 * 2, 192, kernel_size=1)
        self.block2_1 = Block(vis=True, hidden_size=192, mlp_dim=192*4, num_attention_heads=6)
        self.block2_2 = Block(vis=True, hidden_size=192, mlp_dim=192*4, num_attention_heads=6)

        for key in list(checkpoint.keys()):
            if key in unexpected or 'layers.3' in key:
                del checkpoint[key]
        self.swin_transformer.load_state_dict(checkpoint)

    def forward(self, x):

        fm2=self.p2_ch(x[2])
        B,C,H,W=fm2.shape
        fm2_reshaped = Rearrange('b c h w -> b (h w) c')(fm2)
        fm2_ch = self.p2_sw(x[2])
        fm2_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm2_ch)
        sw2 = self.swin_transformer.layers[1](fm2_ch_reshaped)
        fm2 = self.block2_1(fm2_reshaped,sw2)
        fm2=Rearrange('b (h w) c -> b c h w',h=H,w=W)(fm2)
        sw2 = self.block2_2(sw2,fm2_reshaped)
        sw2 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw2)
        cat2=self.p2_cat(torch.cat((fm2,sw2),dim=1))
        # cat2=self.die2(x[1],cat2)
        cat2=x[1]+cat2
        # Level 2
        fm1_ch = self.p1_sw(sw2)
        fm1_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm1_ch)
        sw1 = self.swin_transformer.layers[0](fm1_ch_reshaped)
        fm1 = self.p1_ch(fm2)
        B, C, H, W = fm1.shape
        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(fm1)
        fm1 = self.block1_1(fm1_reshaped,sw1)
        fm1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(fm1)
        sw1 = self.block1_2(sw1,fm1_reshaped)
        sw1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw1)
        cat1 = self.p1_cat(torch.cat((fm1, sw1), dim=1))
        # cat1 = self.die1(x[0],cat1)
        cat1=x[0]+cat1

        return cat1,cat2
class Encoder_PyramidFeatures1(nn.Module):
    def __init__(self, config, img_size=224, in_channels=3,num_classes=9):
        super().__init__()

        model_path = 'weights/swin_tiny_patch4_window7_224.pth'
        self.swin_transformer = SwinTransformer(img_size, in_chans=3)
        checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
        unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight",
                      "patch_embed.norm.bias",
                      "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
                      "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight",
                      "layers.1.downsample.norm.bias",
                      "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight",
                      "layers.2.downsample.norm.bias",
                      "layers.2.downsample.reduction.weight", "norm.weight", "norm.bias"]
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=4, in_chans=3, embed_dim=96,
            norm_layer=nn.LayerNorm)
        resnet = eval(f"torchvision.models.{config.cnn_backbone}(pretrained={config.resnet_pretrained})")
        self.resnet_layers = nn.ModuleList(resnet.children())[:7]
        self.num_classes=num_classes
        self.p1_sw = nn.Conv2d(256, 96, kernel_size=3,stride=1,padding=1)
        self.p1_ch=nn.Conv2d(config.cnn_pyramid_fm[0], config.swin_pyramid_fm[0], kernel_size=1)
        # self.p1_ch = My_Conv_block1(256, 96, stride=1, dilation=1)
        # self.p1_ch = conv_block(256, 96)
        # self.block1_1 = Block(vis=True, hidden_size=96, mlp_dim=384, num_attention_heads=3)
        # self.block1_2 = Block(vis=True, hidden_size=96, mlp_dim=384, num_attention_heads=3)
        # self.p1_cat = nn.Conv2d(96 * 2, 96, kernel_size=1)

        self.p2 = self.resnet_layers[5]
        self.p2_ch = nn.Conv2d(config.cnn_pyramid_fm[1], config.swin_pyramid_fm[1], kernel_size=1)
        self.p2_sw = nn.Conv2d(96, 192, kernel_size=3,stride=2,padding=1)
        # self.p2_ch = My_Conv_block1(96, 192, stride=2, dilation=1)
        # self.p2_ch = conv_block(96, 192,stride=2)
        # self.p2_cat = nn.Conv2d(192 * 2, 192, kernel_size=1)
        # self.block2_1 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=6)
        # self.block2_2 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=6)

        # self.p3 = self.resnet_layers[6]
        self.p3 = self.resnet_layers[6]
        self.p3_ch = nn.Conv2d(config.cnn_pyramid_fm[2],384, kernel_size=1)
        self.p3_sw = nn.Conv2d(192, 384,  kernel_size=3,stride=2,padding=1)
        # self.p3_ch = My_Conv_block1(192, 384, stride=2, dilation=1)
        # self.p3_ch = conv_block(192, 384, stride=2)
        # self.p3_cat = nn.Conv2d(384 * 2, 384, kernel_size=1)
        # self.block3_1 = Block(vis=True, hidden_size=384, mlp_dim=384 * 4, num_attention_heads=12)
        # self.block3_2 = Block(vis=True, hidden_size=384, mlp_dim=384 * 4, num_attention_heads=12)

        for key in list(checkpoint.keys()):
            if key in unexpected or 'layers.3' in key:
                del checkpoint[key]
        self.swin_transformer.load_state_dict(checkpoint)

    def forward(self, x):
        img=x
        for i in range(5):
            x = self.resnet_layers[i](x)
            # Level 1
        fm1=x
        fm1_dw = self.p1_ch(fm1)
        B, C, H, W = fm1.shape
        fm1_reshaped = Rearrange('b c h w -> b (h w) c')(fm1)
        # fm1_ch = self.p1_sw(x)
        # fm1_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm1_ch)
        fm1_ch_reshaped=self.patch_embed(img)
        sw1 = self.swin_transformer.layers[0](fm1_ch_reshaped)
        # sw1_1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw1)
        # draw_features(4, 4, sw1_1.detach().numpy(), "{}/sw1.png".format(savepath))
        # fm1 = self.block1_1(fm1_reshaped, sw1)
        # fm1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(fm1)
        # sw1 = self.block1_2(sw1, fm1_reshaped)
        sw1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw1)


        # Level 2
        fm2_ch = self.p2_sw(sw1)
        fm2_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm2_ch)
        sw2 = self.swin_transformer.layers[1](fm2_ch_reshaped)
        fm2=self.p2(fm1)
        fm2_dw = self.p2_ch(fm2)
        # draw_features(4, 4, fm2.detach().cpu().numpy(), "{}/fm2.png".format(savepath))
        B, C, H, W = fm2.shape
        # fm2_reshaped = Rearrange('b c h w -> b (h w) c')(fm2)
        # fm2 = self.block2_1(fm2_reshaped, sw2)
        # fm2 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(fm2)
        # sw2 = self.block2_2(sw2, fm2_reshaped)
        sw2 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw2)

        # Level 3
        fm3_ch = self.p3_sw(sw2)
        fm3_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm3_ch)
        sw3 = self.swin_transformer.layers[2](fm3_ch_reshaped)
        fm3=self.p3(fm2)
        fm3_dw = self.p3_ch(fm3)
        # draw_features(4, 4, fm3.detach().cpu().numpy(), "{}/fm3.png".format(savepath))
        B, C, H, W = fm3.shape
        # fm3_reshaped = Rearrange('b c h w -> b (h w) c')(fm3)
        # fm3 = self.block3_1(fm3_reshaped, sw3)
        # fm3 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(fm3)
        # sw3 = self.block3_2(sw3, fm3_reshaped)
        sw3 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw3)

        return sw1,fm1_dw,sw2,fm2_dw,sw3,fm3_dw
# class Encoder_PyramidFeatures1_1(nn.Module):
#     def __init__(self, config, img_size=224, in_channels=3):
#         super().__init__()
#
#         model_path = config.swin_pretrained_path
#         self.swin_transformer = SwinTransformer(img_size, in_chans=3)
#         checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
#         unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight",
#                       "patch_embed.norm.bias",
#                       "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
#                       "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight",
#                       "layers.1.downsample.norm.bias",
#                       "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight",
#                       "layers.2.downsample.norm.bias",
#                       "layers.2.downsample.reduction.weight", "norm.weight", "norm.bias"]
#
#         resnet = eval(f"torchvision.models.{config.cnn_backbone}(pretrained={config.resnet_pretrained})")
#         self.resnet_layers = nn.ModuleList(resnet.children())[:7]
#
#         self.p1_sw = nn.Conv2d(256, 96, kernel_size=1)
#         self.p1_ch = conv_block(256, 96)
#         self.block1_1 = Block(vis=True, hidden_size=96, mlp_dim=384, num_attention_heads=3)
#         self.block1_2 = Block(vis=True, hidden_size=96, mlp_dim=384, num_attention_heads=3)
#         self.p1_cat = nn.Conv2d(96 * 2, 96, kernel_size=1)
#
#         self.p2_sw = nn.Conv2d(96, 192, kernel_size=3, stride=2,padding=1)
#         self.p2_ch = conv_block(96, 192, stride=2)
#         self.p2_cat = nn.Conv2d(192 * 2, 192, kernel_size=1)
#         self.block2_1 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=6)
#         self.block2_2 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=6)
#
#         # self.p3 = self.resnet_layers[6]
#         self.p3_sw = nn.Conv2d(192, 384, kernel_size=3, stride=2,padding=1)
#         self.p3_ch = conv_block(192, 384, stride=2)
#         self.p3_cat = nn.Conv2d(384 * 2, 384, kernel_size=1)
#         self.block3_1 = Block(vis=True, hidden_size=384, mlp_dim=384 * 4, num_attention_heads=12)
#         self.block3_2 = Block(vis=True, hidden_size=384, mlp_dim=384 * 4, num_attention_heads=12)
#
#         for key in list(checkpoint.keys()):
#             if key in unexpected or 'layers.3' in key:
#                 del checkpoint[key]
#         self.swin_transformer.load_state_dict(checkpoint)
#
#     def forward(self, x):
#
#         for i in range(5):
#             x = self.resnet_layers[i](x)
#
#             # Level 1
#         fm1 = self.p1_ch(x)
#         B, C, H, W = fm1.shape
#         fm1_reshaped = Rearrange('b c h w -> b (h w) c')(fm1)
#         fm1_ch = self.p1_sw(x)
#         fm1_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm1_ch)
#         sw1 = self.swin_transformer.layers[0](fm1_ch_reshaped)
#         sum=fm1_reshaped + sw1
#         fm1 = self.block1_1(sum, fm1_reshaped)
#         fm1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(fm1)
#         sw1 = self.block1_2(sum, sw1)
#         sw1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw1)
#         cat1 = torch.cat((fm1, sw1), dim=1)
#         cat1 = self.p1_cat(cat1)
#
#         # Level 2
#         fm2_ch = self.p2_sw(cat1)
#         fm2_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm2_ch)
#         sw2 = self.swin_transformer.layers[1](fm2_ch_reshaped)
#         fm2 = self.p2_ch(cat1)
#         B, C, H, W = fm2.shape
#         fm2_reshaped = Rearrange('b c h w -> b (h w) c')(fm2)
#         sum=fm2_reshaped + sw2
#         fm2 = self.block2_1(sum, fm2_reshaped)
#         fm2 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(fm2)
#         sw2 = self.block2_2(sum, sw2)
#         sw2 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw2)
#         cat2 = torch.cat((fm2, sw2), dim=1)
#         cat2 = self.p2_cat(cat2)
#
#         # Level 3
#         fm3_ch = self.p3_sw(cat2)
#         fm3_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm3_ch)
#         sw3 = self.swin_transformer.layers[2](fm3_ch_reshaped)
#         fm3 = self.p3_ch(cat2)
#         B, C, H, W = fm3.shape
#         fm3_reshaped = Rearrange('b c h w -> b (h w) c')(fm3)
#         sum=fm3_reshaped + sw3
#         fm3 = self.block3_1(sum, fm3_reshaped)
#         fm3 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(fm3)
#         sw3 = self.block3_2(sum, sw3)
#         sw3 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw3)
#         cat3 = torch.cat((fm3, sw3), dim=1)
#         cat3 = self.p3_cat(cat3)
#
#         return cat1,cat2,cat3
# class Dncoder_PyramidFeatures1_1(nn.Module):
#     def __init__(self, config, img_size=224, in_channels=3):
#         super().__init__()
#
#         model_path = config.swin_pretrained_path
#         self.swin_transformer = SwinTransformer(img_size, in_chans=3)
#         checkpoint = torch.load(model_path, map_location=torch.device(device))['model']
#         unexpected = ["patch_embed.proj.weight", "patch_embed.proj.bias", "patch_embed.norm.weight",
#                       "patch_embed.norm.bias",
#                       "head.weight", "head.bias", "layers.0.downsample.norm.weight", "layers.0.downsample.norm.bias",
#                       "layers.0.downsample.reduction.weight", "layers.1.downsample.norm.weight",
#                       "layers.1.downsample.norm.bias",
#                       "layers.1.downsample.reduction.weight", "layers.2.downsample.norm.weight",
#                       "layers.2.downsample.norm.bias",
#                       "layers.2.downsample.reduction.weight", "norm.weight", "norm.bias"]
#
#         self.sa2 = SpatialAttention(in_channel=192)
#         self.sa1 = SpatialAttention(in_channel=96)
#
#         self.p1_sw = nn.ConvTranspose2d(in_channels=192, out_channels=96, kernel_size=3,
#                                         stride=2, padding=1, output_padding=1)
#         self.p1_ch = conv_block_decoder(192, 96, stride=2)
#         self.p1_cat = nn.Conv2d(96 * 2, 96, kernel_size=1)
#         self.block1_1 = Block(vis=True, hidden_size=96, mlp_dim=384, num_attention_heads=3)
#         self.block1_2 = Block(vis=True, hidden_size=96, mlp_dim=384, num_attention_heads=3)
#
#         self.p2_sw = nn.ConvTranspose2d(in_channels=384, out_channels=192, kernel_size=3,
#                                         stride=2, padding=1, output_padding=1)
#         self.p2_ch = conv_block_decoder(384, 192, stride=2)
#         self.p2_cat = nn.Conv2d(192 * 2, 192, kernel_size=1)
#         self.block2_1 = Block(vis=True, hidden_size=192, mlp_dim=192*4, num_attention_heads=6)
#         self.block2_2 = Block(vis=True, hidden_size=192, mlp_dim=192*4, num_attention_heads=6)
#
#         for key in list(checkpoint.keys()):
#             if key in unexpected or 'layers.3' in key:
#                 del checkpoint[key]
#         self.swin_transformer.load_state_dict(checkpoint)
#
#     def forward(self, x):
#
#         fm2=self.p2_ch(x[2])
#         B,C,H,W=fm2.shape
#         fm2_reshaped = Rearrange('b c h w -> b (h w) c')(fm2)
#         fm2_ch = self.p2_sw(x[2])
#         fm2_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm2_ch)
#         sw2 = self.swin_transformer.layers[1](fm2_ch_reshaped)
#         sum=fm2_reshaped+sw2
#         fm2 = self.block2_1(sum,fm2_reshaped)
#         fm2=Rearrange('b (h w) c -> b c h w',h=H,w=W)(fm2)
#         sw2 = self.block2_2(sum,sw2)
#         sw2 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw2)
#         cat2=self.p2_cat(torch.cat((fm2,sw2),dim=1))
#         cat2=self.sa2(x[1])*cat2+cat2
#
#         # Level 2
#         fm1_ch = self.p1_sw(cat2)
#         fm1_ch_reshaped = Rearrange('b c h w -> b (h w) c')(fm1_ch)
#         sw1 = self.swin_transformer.layers[0](fm1_ch_reshaped)
#         fm1 = self.p1_ch(cat2)
#         B, C, H, W = fm1.shape
#         fm1_reshaped = Rearrange('b c h w -> b (h w) c')(fm1)
#         sum=fm1_reshaped+sw1
#         fm1 = self.block1_1(sum,fm1_reshaped)
#         fm1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(fm1)
#         sw1 = self.block1_2(sum,sw1)
#         sw1 = Rearrange('b (h w) c -> b c h w', h=H, w=W)(sw1)
#         cat1 = self.p1_cat(torch.cat((fm1, sw1), dim=1))
#         cat1 = self.sa1(x[0])*cat1+cat1
#
#         return cat1,cat2


class Attention1(nn.Module):
    def __init__(self, vis, hidden_size=64, num_attention_heads=2):
        super(Attention1, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)  # 768/12=64
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 12*64=768

        self.query = nn.Linear(hidden_size, self.all_head_size)  # wm,768->768，Wq矩阵为（768,768）
        self.key = nn.Linear(hidden_size, self.all_head_size)  # wm,768->768,Wk矩阵为（768,768）
        self.value = nn.Linear(hidden_size, self.all_head_size)  # wm,768->768,Wv矩阵为（768,768）
        self.out = nn.Linear(hidden_size, hidden_size)  # wm,768->768
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads, self.attention_head_size)  # wm,(bs,197)+(12,64)=(bs,197,12,64)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # wm,(bs,12,197,64)

    def forward(self, hidden_states, qx):
        # hidden_states为：(bs,197,768)
        mixed_query_layer = self.query(qx)  # wm,768->768
        mixed_key_layer = self.key(hidden_states)  # wm,768->768
        mixed_value_layer = self.value(hidden_states)  # wm,768->768

        query_layer = self.transpose_for_scores(mixed_query_layer)  # wm，(bs,12,197,64)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # 将q向量和k向量进行相乘（bs,12,197,197)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 将结果除以向量维数的开方
        attention_probs = self.softmax(attention_scores)  # 将得到的分数进行softmax,得到概率
        weights = attention_probs if self.vis else None  # wm,实际上就是权重
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)  # 将概率与内容向量相乘
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # wm,(bs,197)+(768,)=(bs,197,768)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights  # wm,(bs,197,768),(bs,197,197)


class Mlp(nn.Module):
    def __init__(self, hidden_size=64, mlp_dim=256):
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(self.hidden_size, self.mlp_dim)  # wm,786->3072
        self.fc2 = nn.Linear(self.mlp_dim, self.hidden_size)  # wm,3072->786
        self.act_fn = torch.nn.functional.gelu  # wm,激活函数
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)  # wm,786->3072
        x = self.act_fn(x)  # 激活函数
        x = self.dropout(x)  # wm,丢弃
        x = self.fc2(x)  # wm3072->786
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, vis, hidden_size=64, mlp_dim=256, num_attention_heads=2):
        super(Block, self).__init__()
        self.hidden_size = hidden_size  # wm,768
        self.mlp_dim = mlp_dim
        self.num_attention_heads = num_attention_heads
        self.attention_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)  # wm，层归一化
        self.ffn_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

        self.ffn = Mlp(self.hidden_size, self.mlp_dim)
        self.attn = Attention1(vis, self.hidden_size, self.num_attention_heads)

    def forward(self, x, qx):
        h = x
        x = self.attention_norm(x)
        qx = self.attention_norm(qx)
        x, weights = self.attn(x, qx)
        x = x + h  # 残差结构

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h  # 残差结构
        return x

    def load_from(self, weights, n_block):
            ROOT = f"Transformer/encoderblock_{n_block}/"
            with torch.no_grad():
                query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size,
                                                                                       self.hidden_size).t()
                key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size,
                                                                                     self.hidden_size).t()
                value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size,
                                                                                       self.hidden_size).t()
                out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size,
                                                                                       self.hidden_size).t()

                query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
                key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
                value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
                out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

                self.attn.query.weight.copy_(query_weight)
                self.attn.key.weight.copy_(key_weight)
                self.attn.value.weight.copy_(value_weight)
                self.attn.out.weight.copy_(out_weight)
                self.attn.query.bias.copy_(query_bias)
                self.attn.key.bias.copy_(key_bias)
                self.attn.value.bias.copy_(value_bias)
                self.attn.out.bias.copy_(out_bias)

                mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
                mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
                mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
                mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

                self.ffn.fc1.weight.copy_(mlp_weight_0)
                self.ffn.fc2.weight.copy_(mlp_weight_1)
                self.ffn.fc1.bias.copy_(mlp_bias_0)
                self.ffn.fc2.bias.copy_(mlp_bias_1)

                self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
                self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
                self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
                self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class DLF(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, embed_dim=(96, 192, 384), depth=8, norm_layer=nn.LayerNorm,
                 bilinear=True):
        super().__init__()
        self.cross_pos_embed = config.cross_pos_embed
        self.depth = depth
        self.pyramid = PyramidFeatures(config=config, img_size=img_size, in_channels=in_chans)
        # factor = 2 if bilinear else 1
        # self.scale = 4  # 1 2 4
        # self.global_attention=Transformer_block_global((96, 192, 384), 192, img_size // 4, depth=6, patch_size=1,heads=4)
        self.block1 = Block(vis=True, hidden_size=384, mlp_dim=512, num_attention_heads=8)

        # n_p1 = (config.image_size // config.patch_size     ) ** 2  # default: 3136
        # n_p2 = (config.image_size // config.patch_size // 2) ** 2  # default: 196
        n_p3 = (config.image_size // config.patch_size // 4) ** 2
        # num_patches = (n_p1, n_p2,n_p3)
        # self.num_branches = 3
        self.sa = SpatialAttention(in_channel=1024)
        self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, n_p3, embed_dim[2]))])

        # total_depth = sum([sum(x[-2:]) for x in config.depth])
        # dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, total_depth)]  # stochastic depth decay rule
        # dpr_ptr = 0
        # self.blocks = nn.ModuleList()
        # for idx, block_config in enumerate(config.depth):
        #     curr_depth = max(block_config[:-1]) + block_config[-1]
        #     dpr_ = dpr[dpr_ptr:dpr_ptr + curr_depth]
        #     blk = DLFBlock(embed_dim, num_patches, block_config, num_heads=config.num_heads, mlp_ratio=config.mlp_ratio,
        #                           qkv_bias=config.qkv_bias, qk_scale=config.qk_scale, drop=config.drop_rate,
        #                           attn_drop=config.attn_drop_rate, drop_path=dpr_, norm_layer=norm_layer)
        #     dpr_ptr += curr_depth
        #     self.blocks.append(blk)

        # self.norm = nn.ModuleList([norm_layer(embed_dim[i]) for i in range(self.num_branches)])

        if self.pos_embed[0].requires_grad:
            trunc_normal_(self.pos_embed[0], std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        out = {'cls_token'}
        if self.pos_embed[0].requires_grad:
            out.add('pos_embed')
        return out

    def forward(self, x):
        global GL_Crose
        xs = self.pyramid(x)
        _, _, H, W = xs[2].size()

        if self.cross_pos_embed:
            xs[-1] += self.pos_embed[0]
        for i in range(self.depth):
            if i == 0:
                GL_Crose = self.block1(xs[-1], xs[3])
            else:
                GL_Crose = self.block1(GL_Crose, xs[3])

        GL_Crose = Rearrange('b (h w) c -> b c h w', h=H, w=W)(GL_Crose)
        GL_Crose = self.sa(xs[2]) * GL_Crose + GL_Crose

        # for blk in self.blocks:
        #     xs = blk(xs)
        # xs=self.global_attention(xs[0],xs[1],xs[2])
        # xs = [self.norm[i](x) for i, x in enumerate(xs)]

        return xs[0], xs[1], GL_Crose


