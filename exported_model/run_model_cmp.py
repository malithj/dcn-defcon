#!/usr/bin/python
import argparse
import torch
import torch.nn as nn
import torchprof
import numpy as np
import pandas as pd

import DConv
import DConv2

"""
Interval Search vs SOA Comparison
"""


def parse_layers(mode, technique='is'):
    df = pd.read_csv('exported_model/exported_weights/reb_compare.csv')
    layers = []
    layer_dims = []
    for index, row in df.iterrows():
        is_in_planes, is_out_planes, is_in_height, is_in_width = row['IS_Dimensions'].lstrip(
            '[').rstrip(']').split(',')
        is_stride, is_kernel, is_pad, is_deformable_groups = row['IS_Stride'], row[
            'IS_Kernel'], row['IS_Padding'], row['IS_Deformable_Groups']
        base_in_planes, base_out_planes, base_in_height, base_in_width = row['Base_Dimensions'].lstrip(
            '[').rstrip(']').split(',')
        base_stride, base_kernel, base_pad, base_deformable_groups = row['Base_Stride'], row[
            'Base_Kernel'], row['Base_Padding'], row['Base_Deformable_Groups']
        layer_type = 'DConv'
        if technique == 'is':
            if row['IS_Name'].split(' ')[0].strip() == 'DConv:':
                layers.append(DConv2.DConv(int(is_in_planes), int(is_out_planes), int(
                    is_kernel), int(is_stride), int(is_pad), False, mode))
                layer_type = 'DConv'
            if row['IS_Name'].split(' ')[0].strip() == 'Conv2d:':
                layers.append(nn.Conv2d(int(is_in_planes), int(is_out_planes), int(
                    is_kernel), int(is_stride), int(is_pad), bias=False))
                layer_type = 'Conv2d'
            layer_dims.append([int(is_in_planes), int(is_out_planes), int(is_in_height), int(is_in_width), int(
                is_kernel), int(is_stride), int(is_pad), layer_type])
        else:
            if row['Base_Name'].split(' ')[0].strip() == 'DCN:':
                layers.append(DConv2.DConv(int(base_in_planes), int(base_out_planes), int(
                    base_kernel), int(base_stride), int(base_pad), False, mode))
                layer_type = 'DConv'
            if row['Base_Name'].split(' ')[0].strip() == 'Conv2d:':
                layers.append(nn.Conv2d(int(base_in_planes), int(base_out_planes), int(
                    base_kernel), int(base_stride), int(base_pad), bias=False))
                layer_type = 'Conv2d'
            layer_dims.append([int(base_in_planes), int(base_out_planes), int(base_in_height), int(base_in_width), int(
                base_kernel), int(base_stride), int(base_pad), layer_type])
    return layers, layer_dims


def extract_compute(trace, torch_prof_events_list, layer_idx):
    paths = [trace[x].path for x in layer_idx]
    cuda_times = {}
    for path in paths:
        cuda_time = 0
        for x in torch_prof_events_list[path]:
            cuda_time += x[0].cuda_time
        cuda_time = cuda_time / 1000.0            # convert to milliseconds
        key = tuple([x for x in path])
        format = '%s_%s' if len(key) == 2 else '%s'
        if key in cuda_times:
            cuda_times[format % key] += cuda_time
        else:
            cuda_times[format % key] = cuda_time
    return cuda_times


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer',
                        default=0,
                        type=int,
                        help='layer idx')

    parser.add_argument('--batch',
                        default=1,
                        type=int,
                        help='batch')

    parser.add_argument('--height',
                        default=224,
                        type=int,
                        help='height')

    parser.add_argument('--width',
                        default=224,
                        type=int,
                        help='width')

    parser.add_argument('--itr',
                        default=100,
                        type=int,
                        help='iterations')

    parser.add_argument('--mode',
                        default='ours',
                        type=str,
                        help='mode (ours / torch)')

    parser.add_argument('--net_mode',
                        default='full',
                        type=str,
                        help='network mode')

    parser.add_argument('--profile',
                        action='store_true',
                        help='profile layer')

    args, _ = parser.parse_known_args()
    return args


def main():
    global_df = pd.DataFrame()
    initial = True
    itr = 100
    for idx in range(14):
        for type_ in ['base', 'is']:
            for mode in ['torch', 'ours']:
                layers, layer_dims = parse_layers(mode, type_)
                layer = layers[idx]
                layer.cuda()
                in_height = layer_dims[idx][2]
                in_width = layer_dims[idx][3]
                with torchprof.Profile(layer, use_cuda=True, profile_memory=True) as prof:
                    for i in range(itr):
                        input = torch.rand(
                            [1, layer_dims[idx][0], in_height, in_width]).cuda()
                        layer(input)
                trace, event_lists_dict = prof.raw()
                # print(trace)
                indices = [1, 2] if layer_dims[idx][-1] == 'DConv' else [0]
                cuda_times = extract_compute(trace, event_lists_dict, indices)
                # print(prof.display(show_events=False))
                store_dict = {'OPERATION': list(
                    cuda_times.keys()), 'TIME(ms)': np.asarray(list(cuda_times.values())) / itr}
                df = pd.DataFrame.from_dict(store_dict)
                df['MODE'] = mode
                b = 1
                c = layer_dims[idx][0]
                f = layer_dims[idx][1]
                in_height = layer_dims[idx][2]
                in_width = layer_dims[idx][3]
                df['BATCH'] = b
                df['IN_CHANNELS'] = c
                df['OUT_CHANNELS'] = f
                df['HEIGHT'] = in_height
                df['WIDTH'] = in_width
                df['LAYER_IDX'] = idx
                df['TYPE'] = type_
                if initial:
                    global_df = df
                    initial = False
                else:
                    global_df = global_df.append(df)
    print(global_df)
    global_df.to_csv('results/is_base_cmp_network.csv')


if __name__ == '__main__':
    main()
