import argparse
import numpy as np
import torch
#import torchprof
import pandas as pd

import DConv
import DConv2


def parse_layers(mode, net_mode):
    f = open('exported_model/exported_weights/%s_weight_layer_dims.txt' % net_mode)
    layers = []
    layer_dims = []
    layer_idx = 0
    if net_mode == 'light':
        for line in f:
            if 'layer' in line:
                layer_idx = 0
            if layer_idx == 1:
                splitted = line.split(',')
                inplanes, out_channels, kernel, stride, pad, groups = [
                    x.strip('\n()').split('=')for x in splitted]
                kernel_h, kernel_w = kernel[1].strip('(').split('_')
                stride_h, stride_w = stride[1].strip('(').split('_')
                pad_h, pad_w = pad[1].strip('(').split('_')
            if layer_idx == 4:
                splitted = line.split(',')
                in_channels, planes, kernel, stride, pad, groups, dilation, deformable_groups = [
                    x.strip('\n()').split('=')for x in splitted]
                kernel_h, kernel_w = kernel[1].strip('(').split('_')
                stride_h, stride_w = stride[1].strip('(').split('_')
                pad_h, pad_w = pad[1].strip('(').split('_')
                layers.append(DConv.DConv(int(inplanes[1]), int(planes[1]), int(
                    kernel_h), int(stride_h), int(pad_h), False, mode))
                layer_dims.append([int(inplanes[1]), int(planes[1]), int(
                    kernel_h), int(stride_h), int(pad_h)])
            layer_idx += 1
    else:
        for line in f:
            if 'layer' in line:
                layer_idx = 0
            if layer_idx == 1:
                splitted = line.split(',')
                inplanes, out_channels, kernel, stride, pad, groups = [
                    x.strip('\n()').split('=')for x in splitted]
                kernel_h, kernel_w = kernel[1].strip('(').split('_')
                stride_h, stride_w = stride[1].strip('(').split('_')
                pad_h, pad_w = pad[1].strip('(').split('_')
            if layer_idx == 2:
                splitted = line.split(',')
                in_channels, planes, kernel, stride, pad, groups, dilation, deformable_groups = [
                    x.strip('\n()').split('=')for x in splitted]
                kernel_h, kernel_w = kernel[1].strip('(').split('_')
                stride_h, stride_w = stride[1].strip('(').split('_')
                pad_h, pad_w = pad[1].strip('(').split('_')
                layers.append(DConv2.DConv(int(inplanes[1]), int(planes[1]), int(
                    kernel_h), int(stride_h), int(pad_h), False, mode))
                layer_dims.append([int(inplanes[1]), int(planes[1]), int(
                    kernel_h), int(stride_h), int(pad_h)])
            layer_idx += 1
    return layers, layer_dims


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


def extract_compute(trace, layer_idx):
    paths = [trace[x].path for x in layer_idx]
    cuda_times = {}
    for path in paths:
        cuda_time = 0
        for x in torch_prof_events_list[path]:
            cuda_time += x[0].cuda_time
        cuda_time = cuda_time / 1000.0            # convert to milliseconds
        if (path[0], path[1]) in cuda_times:
            cuda_times[(path[0], path[1])] += cuda_time
        else:
            cuda_times[(path[0], path[1])] = cuda_time
    return cuda_times


def main():
    args = parse_args()
    idx = args.layer
    layers, layer_dims = parse_layers(args.mode, args.net_mode)
    layer = layers[idx]
    layer.load_state_dict(torch.load(
        'exported_model/exported_weights/%s_dcn_%d.pt' % ('search_light' if args.net_mode == 'light' else args.net_mode, (idx + 1))))
    layer.cuda()
    if args.profile:
        #with torchprof.Profile(layer, use_cuda=True, profile_memory=True) as prof:
        with torch.profiler.profile(use_cuda=True,
                                    profile_memory=True) as prof:
            for i in range(args.itr):
                input_ = torch.rand(
                    [1, layer_dims[idx][0], args.height, args.width]).cuda()
                layer(input_)
        trace = prof.key_averages().table(sort_by='self_cuda_time_total')
        # trace, event_lists_dict = prof.raw()
        # cuda_times = extract_compute(
        #    trace, event_lists_dict, [2, 3, 4, 5, 6]) if args.net_mode == 'light' else extract_compute(trace, event_lists_dict, [1, 2])
        # cuda_times = extract_compute(trace, [2, 3, 4, 5, 6]) if args.net_mode == 'light' else extract_compute(trace, [1, 2])
        # print(prof.display(show_events=False))
        # store_dict = {'OPERATION': list(
        #     cuda_times.keys()), 'TIME(ms)': np.asarray(list(cuda_times.values())) / args.itr}
        # df = pd.DataFrame.from_dict(store_dict)
        # df.to_csv('results/%s_runtimes.csv' % args.net_mode)
        # print(df)
    else:
        for i in range(args.itr):
            input = torch.rand(
                [1, layer_dims[idx][0], args.height, args.width]).cuda()
            layer(input)


if __name__ == '__main__':
    main()
