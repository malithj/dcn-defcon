import torch
import torch.nn as nn

from torch.profiler import record_function

from dcn import DeformConv
from torch_dcn import TorchDeformConv


class DConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False, mode='ours'):
        super(DConv, self).__init__()
        with record_function("DConv_conv1"):
            self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                   stride=stride, padding=padding, bias=bias)
            # self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if mode == 'ours':
            with record_function("DConv_conv2"):
                self.conv2 = DeformConv(
                    inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding,  threshold=7)
        else:
            with record_function("DConv_conv2"):
                self.conv2 = TorchDeformConv(
                    inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding)

        self.conv1.weight.data.fill_(0.0)

    def forward(self, x):
        offset = self.conv1(x)
        mask = torch.zeros([offset.size(0), self.conv2.deformable_groups *
                            self.conv2.kernel_size[0] * self.conv2.kernel_size[1], offset.size(2), offset.size(3)]).cuda()
        out = self.conv2(x, offset, mask)
        # out = self.conv2(x, offset)
        return out
