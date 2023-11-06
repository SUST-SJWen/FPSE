import os
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from einops.layers.torch import Rearrange
from thop import profile, clever_format
from torch.backends import cudnn
from torchvision.transforms import transforms

# from datasets.dataset_synapse import augment_seg
from . import HiFormer_configs as configs
from .Encoder2 import DLF, Encoder_PyramidFeatures1_1, Dncoder_PyramidFeatures1_1, draw_features, \
    Encoder_PyramidFeatures1
from .Decoder1 import ConvUpsample, SegmentationHead
# from .cenet import CE_Net_
# from .vision_transformer import SwinUnet
# from .unet import Unet


class SpatialAttention(nn.Module):
    def __init__(self, in_channel=512):
        super(SpatialAttention, self).__init__()

        # assert  kernel_size in (3,7), 'kernel size must be 3 or 7'
        # padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(in_channel, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out_c = torch.mean(x, dim=1, keepdim=True)
        avg_out_s = torch.mean(x, dim=2, keepdim=True)
        avg_out_s = torch.mean(avg_out_s, dim=3, keepdim=True)
        avg_out_s = avg_out_s.repeat(1, 1, x.size()[2], x.size()[3])
        x = avg_out_s + x
        x = self.conv1(x)
        return self.sigmoid(x)


class HiFormer(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 16]
        self.n_classes = n_classes
        self.DLF = DLF(config=config, img_size=img_size, in_chans=in_chans)
        self.ConvUp1 = ConvUpsample(in_chans=384, out_chans=[512], upsample=True)
        self.ConvUp2 = ConvUpsample(in_chans=512, out_chans=[256], upsample=True)

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=1,
        )

        self.sa1 = SpatialAttention(in_channel=512)
        self.sa2 = SpatialAttention(in_channel=256)
        self.sa3 = SpatialAttention(in_channel=3)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                256, 16,
                kernel_size=3, stride=1,
                padding=1, bias=True),
            # nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        xs = self.DLF(x)
        C = self.ConvUp1(xs[2])
        C = self.sa1(xs[1]) * C + C
        C = self.ConvUp2(C)
        C = self.sa2(xs[0]) * C + C
        C = self.conv_pred(C)
        # C=self.sa3(x)*C+C

        out = self.segmentation_head(C)

        return nn.Sigmoid()(out)


class HiFormer1(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 16]
        self.n_classes = n_classes
        self.DLF = DLF3(config=config, img_size=img_size, in_chans=in_chans)
        self.ConvUp1 = ConvUpsample(in_chans=384, out_chans=[192], upsample=True)
        self.ConvUp2 = ConvUpsample(in_chans=192, out_chans=[96], upsample=True)

        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=1,
        )

        self.sa1 = SpatialAttention(in_channel=192)
        self.sa2 = SpatialAttention(in_channel=96)

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                96, 16,
                kernel_size=3, stride=1,
                padding=1, bias=True),
            # nn.GroupNorm(8, 16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        )

    def forward(self, x):
        xs = self.DLF(x)
        C = self.ConvUp1(xs[2])
        C = self.sa1(xs[1]) * C + C
        C = self.ConvUp2(C)
        C = self.sa2(xs[0]) * C + C
        C = self.conv_pred(C)

        out = self.segmentation_head(C)

        return nn.Sigmoid()(out)

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x
class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x


class HiFormer2_1(nn.Module):
    def __init__(self, config=configs.get_hiformer_b_configs(), img_size=224, in_chans=3, num_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 16]
        self.num_classes = num_classes
        self.prim_encoder = Encoder_PyramidFeatures1_1(config=config, img_size=img_size, in_channels=in_chans,num_classes=num_classes)

    def forward(self, x):

        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        sw1,fm1,sw2,fm2,sw3,fm3 = self.prim_encoder(x)
        # dx=self.prim_dncoder(ex)
        # C = self.conv_pred(dx[0])
        # out = self.segmentation_head(C)
        # for i in range(9):
        #     draw_features(1, 1, out[:,i:i+1,:,:].detach().numpy(), "F:/Wen/DAEFormer-main/networks1/features_whitegirl4/myHiFormer/out_"+str(i)+".png".format(savepath))

        return sw1,fm1,sw2,fm2,sw3,fm3
import cv2
# if __name__ == "__main__":
#     import imgaug.augmenters as iaa
#     from datasets.dataset_synapse import Synapse_dataset
#     from torch.utils.data import DataLoader
#     with torch.no_grad():
#         deterministic = 1
#         seed = 1234
#         if not deterministic:
#             cudnn.benchmark = True
#             cudnn.deterministic = False
#         else:
#             cudnn.benchmark = False
#             cudnn.deterministic = True
#
#         random.seed(seed)
#         np.random.seed(seed)
#         torch.manual_seed(seed)
#         torch.cuda.manual_seed(seed)
#
#         CONFIGS = {
#             'hiformer-s': configs.get_hiformer_s_configs(),
#             'hiformer-b': configs.get_hiformer_b_configs(),
#             'hiformer-l': configs.get_hiformer_l_configs(),
#         }
#         print(torch.cuda.is_available())
#         x_transforms = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5])
#         ])
#         y_transforms = transforms.ToTensor()
#         db_train = Synapse_dataset(base_dir="F:/Wen/DAEFormer-main/data/Synapse/train_npz", list_dir='F:\Wen\DAEFormer-main\lists\lists_Synapse', split="train",
#                                    img_size=224,
#                                    norm_x_transform=x_transforms, norm_y_transform=y_transforms)
#         trainloader = DataLoader(db_train, batch_size=1, shuffle=True, num_workers=0,
#                                  pin_memory=True)
#         for i_batch, sampled_batch in enumerate(trainloader):
#             if i_batch==13:
#                 image_batch, label_batch = sampled_batch['image'], sampled_batch['label'].squeeze(1)
#                 image1 = transforms.ToPILImage()(image_batch.squeeze(1))
#                 image1.show()
#                 image1.save('F:/Wen/DAEFormer-main/networks1/features_whitegirl4/image_batch.jpg')
#                 image = transforms.ToPILImage()(label_batch)
#                 image.show()
#                 image.save('F:/Wen/DAEFormer-main/networks1/features_whitegirl4/label_batch.jpg')
#                 # img_x = Image.open("E:/skin/Skin_Cancer_dataset/big_data/test/image/0007.jpg")
#                 # # img_x = img_x.convert('L')
#                 # img_x = img_x.resize((224, 224))
#                 # img_x = x_transforms(img_x)
#                 # img_x = torch.unsqueeze(img_x, 0)
#                 model = HiFormer2_1(config=CONFIGS['hiformer-b'], img_size=224, in_chans=3)
#                 model.load_state_dict(torch.load('F:\Wen\DAEFormer-main\model_out/myHiFormer1_out/best_model.pth', map_location='cpu'))
#                 output=model(image_batch)
#                 model1 = CE_Net_(num_channels=9)
#                 model1.load_state_dict(
#                     torch.load('F:\Wen\DAEFormer-main\model_out/ce_out/best_model.pth', map_location='cpu'))
#                 output1 = model1(image_batch)
#                 model2 = Unet(num_classes=9)
#                 model2.load_state_dict(
#                     torch.load('F:\Wen\DAEFormer-main\model_out/unet_out/best_model.pth', map_location='cpu'))
#                 output2 = model2(image_batch)
#                 model3 = SwinUnet(num_classes=9)
#                 model3.load_state_dict(
#                     torch.load('F:\Wen\DAEFormer-main\model_out/swinUnet_out/best_model.pth', map_location='cpu'))
#                 output3 = model3(image_batch)
#                 model.eval()
#                 model1.eval()
#                 model2.eval()
#                 model3.eval()
#
#                 break
#         # flops1, params1 = profile(model, (img_x,))
#         # flops1, params1 = clever_format([flops1, params1], "%.3f")  # 将数据换算为G以及MB的函数
#         # print(flops1, params1)
#         # torch.save(model.state_dict(), 'U_Net_RAW.pth')