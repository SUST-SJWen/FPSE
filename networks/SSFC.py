import torch
import torch.nn as nn


class SSFC(torch.nn.Module):
    def __init__(self, in_ch, delta=1e-4):
        super(SSFC, self).__init__()

        self.delta = delta
        # self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    def forward(self, x):
        _, _, h, w = x.size()

        q = x.mean(dim=[2, 3], keepdim=True)
        # k = self.proj(x)
        square = (x - q).pow(2)
        sigma = square.sum(dim=[2, 3], keepdim=True) / (h * w - 1)
        att_map = square / (2 * sigma + self.delta) + 0.5
        att_weight = nn.Sigmoid()(att_map)

        return x * att_weight


if __name__ == '__main__':
    torch.manual_seed(1000)
    x = torch.randn(8, 512, 16, 16)
    att = SSFC(512)
    v = att(x)
    print(x, v)
