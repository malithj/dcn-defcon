#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
import torch
from torch import nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.autograd.function import once_differentiable
from torchvision.ops import DeformConv2d

import dcn_ext as _backend


class _TorchDeformConv(Function):
    @staticmethod
    def forward(ctx, input, offset, mask, weight, bias,
                stride, padding, dilation, groups, deformable_groups, use_mask):
        ctx.stride = _pair(stride)
        ctx.padding = _pair(padding)
        ctx.dilation = _pair(dilation)
        ctx.groups = groups
        ctx.deformable_groups = deformable_groups
        ctx.use_mask = use_mask
        output = _backend.torch_warp_dcn_v3_forward(input, weight, offset, mask, bias, ctx.stride[0], ctx.stride[1],
                                                    ctx.padding[0], ctx.padding[1],
                                                    ctx.dilation[0], ctx.dilation[1],
                                                    ctx.groups,
                                                    ctx.deformable_groups, ctx.use_mask)
        ctx.save_for_backward(input, weight, offset, mask, bias)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, weight, offset, mask, bias = ctx.saved_tensors
        grad_input, grad_weight, grad_offset, grad_mask, grad_bias = \
            _backend._torch_warp_dcn_v3_backward(grad_output, input, weight,
                                                 offset, mask, bias,
                                                 ctx.stride[0], ctx.stride[1],
                                                 ctx.padding[0], ctx.padding[1],
                                                 ctx.dilation[0], ctx.dilation[1],
                                                 ctx.groups,
                                                 ctx.deformable_groups,
                                                 ctx.use_mask)

        return grad_input, grad_offset, grad_mask, grad_weight, grad_bias,\
            None, None, None, None, None, None,


torch_dcn_conv = _TorchDeformConv.apply


class TorchDeformConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 num_deformable_groups=1,
                 use_mask=False):
        super(TorchDeformConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = num_deformable_groups
        self.groups = groups
        self.use_mask = use_mask

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size))

        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.fill_(0.1)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return torch_dcn_conv(input, offset, mask,
                              self.weight,
                              self.bias,
                              self.stride,
                              self.padding,
                              self.dilation,
                              self.groups,
                              self.deformable_groups,
                              self.use_mask)


class TorchDCN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 num_deformable_groups=1):
        super(TorchDCN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = num_deformable_groups
        _out_channels = 2 * self.kernel_size[0] * \
            self.kernel_size[1] * self.deformable_groups
        self.conv_offset = nn.Conv2d(
            self.in_channels,
            _out_channels,
            kernel_size=self.kernel_size,
            stride=_pair(self.stride),
            padding=_pair(self.padding),
            bias=False)
        self.deform_conv = TorchDeformConv(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=1,
            num_deformable_groups=self.deformable_groups,
            use_mask=False)
        # self.deform_conv = DeformConv2d(
        # in_channels, out_channels, kernel_size=kernel_size, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=num_deformable_groups, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.conv_offset.weight.data.fill_(0.01)
        self.deform_conv.weight.data.fill_(0.1)

    def forward(self, input):
        offset = self.conv_offset(input)
        out_height, out_width = offset.size()[2:4]
        mask = torch.zeros([offset.size()[0], self.deformable_groups *
                            self.kernel_size[0] * self.kernel_size[1], out_height, out_width]).cuda()
        out = self.deform_conv(input, offset, mask)
        return offset, out
