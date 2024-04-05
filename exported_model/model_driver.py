from itertools import islice
import os
import time

import numpy as np
import pandas as pd


def parse_layers(net_mode):
    f = open('exported_model/exported_weights/%s_weight_layer_dims.txt' % net_mode)
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
                layer_dims.append([int(inplanes[1]), int(planes[1]), int(
                    kernel_h), int(stride_h), int(pad_h)])
            layer_idx += 1
    return layer_dims


def process_stats(file_name, iterations, mode):
    df = pd.read_csv(file_name, delimiter=',',
                     skipinitialspace=True, skiprows=[1])
    unit_line = ""
    with open(file_name) as f:
        for line in islice(f, 1, 2):
            unit_line = line
    splitted = unit_line.split(",")
    time_multiplier = 1 if splitted[2] == 'ms' else 1000
    exec_time = df[df['Type'] == 'GPU activities']['Time'].sum() * \
        time_multiplier / iterations
    deformable_time = df[df['Name'].str.contains(
        'deformable_im2col', regex=False)]['Time'] * time_multiplier / iterations
    return exec_time, deformable_time


def main():
    image_sizes = [(138, 138), (69, 69), (69, 69),
                   (69, 69), (35, 35), (35, 35), (18, 18), (18, 18)]
    batches = [1]
    modes = ['torch', 'ours']
    net_modes = ['full', 'bounded', 'light']
    count = len(batches) * len(image_sizes) * len(modes) * len(net_modes)
    data_ = np.zeros((count, 10))
    prog_iterations = 1000

    for n_idx, net_mode in enumerate(net_modes):
        layers = parse_layers(net_mode)
        for idx, layer in enumerate(layers):
            b = batches[0]
            c = layer[0]
            f = layer[1]
            in_height = image_sizes[idx][0]
            in_width = image_sizes[idx][1]
            for m_idx, m in enumerate(modes):
                idx_ = n_idx * len(layers) * len(modes) + \
                    idx * len(modes) + m_idx
                template_command = "/usr/local/cuda/bin/nvprof --openacc-profiling off --unified-memory-profiling off --log-file 'results/%s.csv' --csv python exported_model/run_model.py --mode %s --in_channels %s --out_channels %s --in_height %s --in_width %s --batch %s --itr %s --net_mode %s --layer %s"
                command = template_command % (
                    'stats', m, c, f, in_height, in_width, b, prog_iterations, net_mode, idx)
                print("running command: ", command)
                ret = os.system(command)
                try:
                    # os.system("sed -i -e 1,4d results/stats.csv")
                    f_ = open("results/stats.csv", "r")
                    f_target = open(
                        "results/tmp_stats_%s_%s.csv" % (str(c), m), "w")
                    count = 0
                    for line in f_.readlines():
                        if line[0:2] != '==':
                            if line[0:6] == '\"Type\"':
                                if count != 1:
                                    count += 1
                                else:
                                    break
                            f_target.write(line)
                    f_.close()
                    f_target.close()
                    exec_time = 0
                    deformable_im2col = 0
                    try:
                        exec_time, deformable_im2col = process_stats(
                            "results/tmp_stats_%s_%s.csv" % (str(c), m), prog_iterations, m)
                    except Exception as process_e:
                        print("error while processing data", process_e)
                except Exception as e:
                    print(
                        "error detected while removing comments on profiled file", e)
                    continue
                data_[idx_, 0] = b
                data_[idx_, 1] = c
                data_[idx_, 2] = f
                data_[idx_, 3] = in_height
                data_[idx_, 4] = in_width
                data_[idx_, 5] = m_idx
                data_[idx_, 6] = n_idx
                data_[idx_, 7] = idx
                data_[idx_, 8] = exec_time
                data_[idx_, 9] = deformable_im2col
    df = pd.DataFrame(
        data_, columns=['BATCH', 'IN_CHANNELS', 'OUT_CHANNELS', 'IN_HEIGHT', 'IN_WIDTH', 'MODE', 'NET_MODE', 'LAYER_IDX', 'TOTAL_TIME(ms)', 'DEFORMABLE_IM2COL(ms)'])
    df['MODE'] = df['MODE'].map(lambda x: modes[int(x)])
    df['NET_MODE'] = df['NET_MODE'].map(lambda x: net_modes[int(x)])
    df.to_csv('results/model_deformable_run_times_%s.csv' %
              (time.time()), index=True)


if __name__ == '__main__':
    main()
