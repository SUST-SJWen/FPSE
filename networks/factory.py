from pathlib import Path
import yaml
import torch
import math
import os
import torch.nn as nn
from PIL import Image
from thop import profile, clever_format
import argparse

from timm.models import swinv2_tiny_window8_256
from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer
from torchvision import models
import torch.nn.functional as F
from torchvision.transforms import transforms

import config1
from vit import VisionTransformer
from utils2 import checkpoint_filter_fn
from decoder import DecoderLinear
from decoder import MaskTransformer
from os.path import join as pjoin
from segmenter import Segmenter, Segmenter_tags2_9201, draw_features, savepath
from utils1 import torch as ptu
import yaml
from pathlib import Path
import os


def load_config():
    return yaml.load(
        open(Path(__file__).parent / "config.yml", "r"), Loader=yaml.FullLoader
    )

@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model


def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")

    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    mlp_expansion_ratio = 4
    model_cfg["d_ff"] = mlp_expansion_ratio * model_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )
    model = VisionTransformer(**model_cfg)
    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        state_dict = torch.load(path, map_location="cpu")
        filtered_dict = checkpoint_filter_fn(state_dict, model)
        model.load_state_dict(filtered_dict, strict=True)
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, default_cfg)

    return model


def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder_cfg["patch_size"] = 16
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")
    return decoder

class Conv(nn.Module):
    def __init__(self, in_channel=150, num_classes=1, zero_head=False, vis=False):
        super(Conv, self).__init__()
        self.conv=nn.Conv2d(in_channel,num_classes,1)
        self.sig=nn.Sigmoid()

    def forward(self,x):
         x=self.conv(x)
         # x=self.sig(x)
         return x

def create_segmenter(num_classes):
    cfg = config1.load_config()
    net_kwargs = cfg["model"]['vit_tiny_patch16_384']
    decoder_cfg = cfg["decoder"]["mask_transformer"]
    net_kwargs["image_size"] = (512, 512)
    net_kwargs["backbone"] = 'vit_tiny_patch16_384'
    net_kwargs["dropout"] = 0.0
    net_kwargs["drop_path_rate"] = 0.1
    decoder_cfg["name"] = 'mask_transformer'
    net_kwargs["decoder"] = decoder_cfg
    net_kwargs["n_cls"] = 150
    model_cfg = net_kwargs.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"],num_classes=num_classes)
    # checkpoint_path = Path(__file__).parent / "weights/checkpoint_tiny.pth"
    # print(f"Resuming training from checkpoint: {checkpoint_path}")
    # checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # model.load_state_dict(checkpoint["model"])
    # models = nn.ModuleList()
    # models.append(model)
    # models.append(Conv())
    return model
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

class My_Segmenter(nn.Module):
    def __init__(self, img_size=224, in_channels=3):
        super().__init__()
        cfg = load_config()
        net_kwargs = cfg["model"]['vit_tiny_patch16_384']
        decoder_cfg = cfg["decoder"]["mask_transformer"]
        net_kwargs["image_size"] = (512, 512)
        net_kwargs["backbone"] = 'vit_tiny_patch16_384'
        net_kwargs["dropout"] = 0.0
        net_kwargs["drop_path_rate"] = 0.1
        decoder_cfg["name"] = 'mask_transformer'
        net_kwargs["decoder"] = decoder_cfg
        net_kwargs["n_cls"] = 150
        self.Transformer = create_segmenter(net_kwargs)
        resnet = models.resnet50(pretrained=True)
        self.resnet_layers = nn.ModuleList(resnet.children())[:7]
        self.block1 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=3)
        self.block2 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=3)
        self.block3 = Block(vis=True, hidden_size=192, mlp_dim=192 * 4, num_attention_heads=3)
        self.conv1x1=nn.Conv2d(256,192,kernel_size=4,stride=4)
        self.Conv1=nn.Conv2d(192,192,kernel_size=3,stride=1,padding=1)
        self.Conv2 = nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1)
        self.LN=nn.Linear(192,150)
    def forward(self, x):
        x_L=x.clone()
        x_G=x.clone()
        for i in range(5):
            x_L = self.resnet_layers[i](x_L)
            # Level 1
        x_L=self.conv1x1(x_L)
        H, W = x.size(2), x.size(3)
        x_G=self.Transformer[0].encoder.patch_embed(x_G)
        x_G=self.Transformer[0].encoder.dropout(x_G)
        for i in range(4):
            x_G = self.Transformer[0].encoder.blocks[i](x_G)
        x_G=self.block1(x_L.flatten(2).transpose(1,2),x_G)
        for i in range(4,8):
            x_G = self.Transformer[0].encoder.blocks[i](x_G)
        x_L=self.Conv1(x_L)
        x_G = self.block2(x_L.flatten(2).transpose(1,2), x_G)
        for i in range(8, 12):
            x_G = self.Transformer[0].encoder.blocks[i](x_G)
        x_L = self.Conv2(x_L)
        x_G = self.block3(x_L.flatten(2).transpose(1, 2), x_G)
        x_G=self.Transformer[0].encoder.norm(x_G)
        # x_G = self.Transformer[0].encoder.head(x_G)
        x_G = self.Transformer[0].encoder.pre_logits(x_G)
        x_G=self.Transformer[0].decoder(x_G, (H, W))
        x_G = F.interpolate(x_G, size=(H, W), mode="bilinear")
        x_G=self.Transformer[1](x_G)
        return x_G
def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"
    with open(variant_path, "r") as f:
        variant = yaml.load(f, Loader=yaml.FullLoader)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)
    data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]

    model.load_state_dict(checkpoint, strict=True)

    return model, variant


if __name__ == "__main__":
    # with torch.no_grad():
    #     input = torch.rand(1, 3, 512, 512)
    #     model = My_Segmenter()
    #     print(model(input).shape)
    #     flops1, params1 = profile(model, (input,))
    #     flops1, params1 = clever_format([flops1, params1], "%.3f")#将数据换算为G以及MB的函数
    #     print(flops1, params1)
    #     #torch.save(model.state_dict(), 'U_Net_RAW.pth')

    with torch.no_grad():
        cfg = load_config()
        net_kwargs = cfg["model"]['vit_tiny_patch16_384']
        decoder_cfg = cfg["decoder"]["mask_transformer"]
        net_kwargs["image_size"] = (512, 512)
        net_kwargs["backbone"] = 'vit_tiny_patch16_384'
        net_kwargs["dropout"] = 0.0
        net_kwargs["drop_path_rate"] = 0.1
        decoder_cfg["name"] = 'mask_transformer'
        net_kwargs["decoder"] = decoder_cfg
        net_kwargs["n_cls"] = 150
        model = create_segmenter(net_kwargs)
        model.load_state_dict(torch.load('weights/weights_best.pth', map_location="cpu"))
        x_transforms = transforms.Compose([
            transforms.ToTensor(),  # 将数据转为tensor类型，方便pytorch进行自动求导，优化之类的操作
            # transforms.Lambda(lambda x: x.repeat(3,1,1)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 数据归一化，两个参数，一个为均值，一个为方差，均设置为0.5，每个参数里三个0.5表示有三个通道
            # transforms.Normalize([0.5], [0.5])  # 单通道
        ])
        img_x = Image.open("E:/skin/Skin_Cancer_dataset/big_data/test/image/0007.jpg")
        # img_x = img_x.convert('L')
        img_x = img_x.resize((512, 512))
        img_x = x_transforms(img_x)
        img_x = torch.unsqueeze(img_x, 0)
        output=model(img_x)
        flops1, params1 = profile(model, (img_x,))
        flops1, params1 = clever_format([flops1, params1], "%.3f")#将数据换算为G以及MB的函数
        print(flops1, params1)
        # torch.save(model.state_dict(), 'U_Net_RAW.pth')