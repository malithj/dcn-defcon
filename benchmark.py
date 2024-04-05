#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division


import argparse
import time
import torch
import torch.nn as nn
from torch.autograd import gradcheck

from dcn import DCN
from torch_dcn import TorchDCN


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--in_channels',
                        default=32,
                        type=int,
                        help='in channels')

    parser.add_argument('--out_channels',
                        default=32,
                        type=int,
                        help='out channels')

    parser.add_argument('--in_height',
                        default=224,
                        type=int,
                        help='in height')

    parser.add_argument('--in_width',
                        default=224,
                        type=int,
                        help='in width')

    parser.add_argument('--batch',
                        default=1,
                        type=int,
                        help='batch')

    parser.add_argument('--itr',
                        default=100,
                        type=int,
                        help='iterations')

    parser.add_argument('--mode',
                        default='ours',
                        type=str,
                        help='mode (ours / torch)')

    args, _ = parser.parse_known_args()
    in_channels = args.in_channels
    out_channels = args.out_channels
    in_height = args.in_height
    in_width = args.in_width
    batch = args.batch
    iterations = args.itr
    # wrap all things (offset and mask) in DCN
    for i in range(iterations):
        input = torch.randn(batch, in_channels, in_height, in_width).cuda()
        if args.mode == 'ours':
            dcn = DCN(in_channels, out_channels, kernel_size=(3, 3), stride=1,
                      padding=1, threshold=2).cuda()
            dcn.eval()
        elif args.mode == 'torch':
            dcn = TorchDCN(in_channels, out_channels, kernel_size=(3, 3),
                           stride=1, padding=1).cuda()
            dcn.eval()
        expected_offset, expected_output = dcn(input)


if __name__ == '__main__':
    main()
