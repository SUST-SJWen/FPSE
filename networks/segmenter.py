import math
import os
import time
from os.path import join as pjoin

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models import swinv2_tiny_window8_256
from torchvision import models

from .utils2 import padding, unpadding
from timm.models.layers import trunc_normal_
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
        mixed_value_layer = self.value(qx)  # wm,768->768

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

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)
ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"
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
import torch
import torch.nn as nn
import math
class CBAM(nn.Module):
    def __init__(self,in_channel,reduction=16,kernel_size=7):
        super(CBAM, self).__init__()
        #通道注意力机制
        self.max_pool=nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(in_features=in_channel,out_features=in_channel//reduction,bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel//reduction,out_features=in_channel,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
        #空间注意力机制
        self.conv=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size ,stride=1,padding=kernel_size//2,bias=False)

    def forward(self,x):
        #通道注意力机制
        maxout=self.max_pool(x)
        maxout=self.mlp(maxout.view(maxout.size(0),-1))
        avgout=self.avg_pool(x)
        avgout=self.mlp(avgout.view(avgout.size(0),-1))
        channel_out=self.sigmoid(maxout+avgout)
        channel_out=channel_out.view(x.size(0),x.size(1),1,1)
        channel_out=channel_out*x
        #空间注意力机制
        max_out,_=torch.max(channel_out,dim=1,keepdim=True)
        mean_out=torch.mean(channel_out,dim=1,keepdim=True)
        out=torch.cat((max_out,mean_out),dim=1)
        out=self.sigmoid(self.conv(out))
        out=out*channel_out
        return out
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
        out = out * x
        return out,out_weight

class DIE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3):
        super(DIE, self).__init__()
        self.kernel_size = kernel_size
        self.inC=inC
        self.outC=outC
        self.ca=Channel_Attention(inC)
        # self.down2 = nn.Conv2d(inC, inC // 4, 1)
        # self.encoder = nn.Conv2d(inC // 4, self.up_factor ** 2 * self.kernel_size ** 2,
        #                          self.kernel_size, 1, self.kernel_size // 2)
        # self.out = nn.Conv2d(inC, outC, kernel_size=4,stride=1)
        self.nn_Unfold = nn.Unfold(kernel_size=(4, 4), dilation=1, padding=0, stride=4)


    def forward(self, in_tensor):
        b,h,w = in_tensor.size(0),in_tensor.size(2),in_tensor.size(3)
        x_L = self.nn_Unfold(in_tensor)
        x_L = x_L.transpose(1, 2)
        x_L = x_L.view(x_L.shape[0], x_L.shape[1], -1, 4, 4)
        G = x_L.size(1)
        x_L_1=x_L.clone()
        # x_L = Rearrange('b l c h w -> (b l) c h w')(x_L)
        for l in range(G):
            a = self.ca(torch.squeeze(x_L[:, l:l+1], 1))
            x_L_1[:, l:l+1]=torch.unsqueeze(a,1)
        # x_L = self.ca(x_L)
        x_L = x_L_1.flatten(2).transpose(1, 2)
        x_L = nn.Fold(output_size=(h, w), kernel_size=(4,4), stride=4)(x_L)
        return x_L
class depthwise_separable_conv(nn.Module):
    def __init__(self, ch_in, ch_out,kernel_size=3):
        super(depthwise_separable_conv, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.depth_conv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, padding=0,stride=kernel_size, groups=ch_in)
        self.point_conv = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class mymodel(nn.Module):
    def __init__(self,inC, outC, kernel_size=3):
        super(mymodel, self).__init__()
        self.conv2d = depthwise_separable_conv(inC, outC,kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)

        return x
import numpy as np
import matplotlib.pyplot as plt
savepath = r'features'
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
    fig.show()
    fig.clf()
    plt.close()
    print("time:{}".format(time.time() - tic))

class Conv_block(nn.Module):
    def __init__(self,ch_in=256,ch_out=96,stride=1,kernel_size=3,dilation=1,padding=1,feather_size=56,pool_kernel_size=4,DE="Down"):
        super(Conv_block,self).__init__()
        self.stride=stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=1, dilation=dilation,
                      padding=dilation, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=kernel_size, stride=stride, dilation=dilation,
                      padding=dilation, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.sa = Spatial_Attention()
        self.ca = Channel_Attention(ch_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x_L_conv=self.conv2(self.conv1(x))
        x_L_ca, out_weight_ca = self.ca(x_L_conv)
        x_L_sa, out_weight_sa = self.sa(x_L_conv)
        all_weight = nn.Sigmoid()(torch.mul(out_weight_ca, out_weight_sa))
        x_L_all = all_weight * x_L_conv
        x_L = x_L_ca + x_L_sa + x_L_all
        return x_L

class conv_block_decoder(nn.Module):
    def __init__(self,ch_in,ch_out,stride=1):
        super(conv_block_decoder,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3,
                               stride=stride, padding=1, output_padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x
class Segmenter(nn.Module):
    def __init__(
        self,
        num_classes
    ):
        super().__init__()
        self.num_classes=num_classes
        self.model_ft = swinv2_tiny_window8_256(pretrained=True)
        resnet = models.resnet50(pretrained=True)
        self.resnet_layers = nn.ModuleList(resnet.children())[:5]
        self.block1_1 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=3)
        self.block1_2 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=3)

        self.block2_1 = Block(vis=True, hidden_size=384, mlp_dim=384 * 4, num_attention_heads=6)
        self.block2_2 = Block(vis=True, hidden_size=384, mlp_dim=384 * 4, num_attention_heads=6)

        self.block3_1 = Block(vis=True, hidden_size=768, mlp_dim=768 * 4, num_attention_heads=8)
        self.block3_2 = Block(vis=True, hidden_size=768, mlp_dim=768 * 4, num_attention_heads=8)
        # self.DIE = DIE(256, 192, kernel_size=4)
        self.conv1 = Conv_block(256,192,stride=2,kernel_size=3,padding=1)
        self.conv2 = Conv_block(192, 384, stride=2, kernel_size=3, padding=1)
        self.conv3 = Conv_block(384, 768, stride=2, kernel_size=3, padding=1)
        self.up_conv1_L=conv_block_decoder(768,384,stride=2)
        self.up_conv2_L = conv_block_decoder(384, 192, stride=2)
        self.up_conv3_L = conv_block_decoder(192, 64, stride=2)
        self.conv_L = nn.Conv2d(64, self.num_classes, 1)

        self.up_conv1_G = conv_block_decoder(768, 384, stride=2)
        self.up_conv2_G = conv_block_decoder(384, 192, stride=2)
        self.up_conv3_G = conv_block_decoder(192, 64, stride=2)
        self.conv_G = nn.Conv2d(64, self.num_classes, 1)

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H,W=im.size(2),im.size(3)
        if im.size()[1] == 1:
            im = im.repeat(1, 3, 1, 1)
        x_L = im.clone()
        for i in range(5):
            x_L = self.resnet_layers[i](x_L)
        x_G = self.model_ft.patch_embed(im)
        x_G = self.model_ft.pos_drop(x_G)
        # draw_features(10, 10, x_L.detach().numpy(), "{}/x.png".format(savepath))
        x_L_1 = self.conv1(x_L)
        h,w=x_L_1.size(2),x_L_1.size(3)
        x_L_1 = Rearrange('b c h w -> b (h w) c')(x_L_1)
        x_G_1 = self.model_ft.layers[0](x_G)
        x_G_1 = self.block1_1(x_L_1, x_G_1)
        x_L_1 = self.block1_2(x_G_1, x_L_1)
        x_L_1 = Rearrange('b (h w) c -> b c h w',h=h,w=w)(x_L_1)

        x_L_2 = self.conv2(x_L_1)
        h, w = x_L_2.size(2), x_L_2.size(3)
        x_L_2 = Rearrange('b c h w -> b (h w) c')(x_L_2)
        x_G_2 = self.model_ft.layers[1](x_G_1)
        x_G_2 = self.block2_1(x_L_2, x_G_2)
        x_L_2 = self.block2_2(x_G_2, x_L_2)
        x_L_2 = Rearrange('b (h w) c -> b c h w', h=h, w=w)(x_L_2)

        x_L_3 = self.conv3(x_L_2)
        h, w = x_L_3.size(2), x_L_3.size(3)
        x_L_3 = Rearrange('b c h w -> b (h w) c')(x_L_3)
        x_G_3 = self.model_ft.layers[2](x_G_2)
        x_G_3 = self.block3_1(x_L_3, x_G_3)
        x_L_3 = self.block3_2(x_G_3, x_L_3)
        x_L_3 = Rearrange('b (h w) c -> b c h w', h=h, w=w)(x_L_3)
        x_G_3 = Rearrange('b (h w) c -> b c h w', h=h, w=w)(x_G_3)

        decoder_conv_L=self.up_conv1_L(x_L_3)
        decoder_conv_L = self.up_conv2_L(decoder_conv_L)
        decoder_conv_L = self.up_conv3_L(decoder_conv_L)

        decoder_conv_G = self.up_conv1_G(x_G_3)
        decoder_conv_G = self.up_conv2_G(decoder_conv_G)
        decoder_conv_G = self.up_conv3_G(decoder_conv_G)

        decoder_conv_L = F.interpolate(decoder_conv_L, size=(H, W), mode="bilinear")
        decoder_conv_G = F.interpolate(decoder_conv_G, size=(H, W), mode="bilinear")
        decoder_conv_L =self.conv_L(decoder_conv_L)
        decoder_conv_G = self.conv_G(decoder_conv_G)
        return decoder_conv_L,decoder_conv_G

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)

class Segmenter_tags2_9201(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        resnet = models.resnet50(pretrained=True)

        self.resnet_layers = nn.ModuleList(resnet.children())[:7]
        self.block1 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=3)
        # self.block2 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=3)
        self.conv1x1 = nn.Conv2d(256, 192, kernel_size=4, stride=4)
        # self.Conv1 = nn.Conv2d(192*2, 192, kernel_size=1)

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        x_L = im.clone()

        for i in range(5):
            x_L = self.resnet_layers[i](x_L)
        x_L = self.conv1x1(x_L)
        h,w=x_L.size(2), x_L.size(3)
        x_L =Rearrange('b c h w -> b (h w) c')(x_L)
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)
        x = self.encoder(im, return_features=True)
        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        fm1 = self.block1(x_L, x)
        # fm2 = self.block2(x, x_L)
        # fm1 = Rearrange('b (h w) c -> b c h w', h=h, w=w)(fm1)
        # fm2 = Rearrange('b (h w) c -> b c h w', h=h, w=w)(fm2)
        # cat = torch.cat((fm1, fm2), dim=1)
        # cat = self.Conv1(cat)
        # cat = Rearrange(' b c h w-> b (h w) c', h=h, w=w)(cat)
        masks = self.decoder(fm1 + x, (H, W))
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
class Segmenter_source(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder


    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder)
        )
        return nwd_params

    def forward(self, im):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]
        masks = self.decoder(x, (H, W))
        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        masks = unpadding(masks, (H_ori, W_ori))

        return masks

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_features=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)

# input = torch.rand(1, 3, 256, 256)
# model_ft = swinv2_tiny_window8_256(pretrained=True)
# o1=model_ft.patch_embed(input)
# o1=model_ft.pos_drop(o1)
# print(o1.size())
# o1=model_ft.layers[0](o1)
# print(o1.size())
# o1=model_ft.layers[1](o1)
# print(o1.size())
# o1=model_ft.layers[2](o1)
# print(o1.size())