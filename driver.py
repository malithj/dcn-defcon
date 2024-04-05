from itertools import islice
import os
import time

import numpy as np
import pandas as pd


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
    layers = [[128, 128, 138, 138], [128, 128, 69, 69],
              [256, 256, 69, 69], [256, 256, 35, 35],
              [512, 512, 35, 35], [512, 512, 18, 18]]
    layer_idx_lookup = [0, 1, 2, 4, 5, 6]
    batches = [1]
    # in_channels = [16, 32, 64, 128, 512, 1024]
    in_channels = [3, 16, 32, 64, 128]
    out_channels = [1]
    modes = ['torch', 'ours']
    count = len(batches) * len(layers) * len(modes)
    data_ = np.zeros((count, 8))
    prog_iterations = 1000
    net_mode = 'bounded'

    for idx, layer in enumerate(layers):
        b_idx = 0
        b = batches[0]
        c = layer[0]
        f = layer[1]
        in_height = layer[2]
        in_width = layer[3]
        for m_idx, m in enumerate(modes):
            idx_ = (idx * len(modes)) + m_idx
            template_command = "/usr/local/cuda/bin/nvprof --openacc-profiling off --unified-memory-profiling off --log-file 'results/%s.csv' --csv python exported_model/run_model.py --mode %s --in_channels %s --out_channels %s --in_height %s --in_width %s --batch %s --itr %s --net_mode %s --layer %s"
            command = template_command % (
                'stats', m, c, f, in_height, in_width, b, prog_iterations, net_mode, layer_idx_lookup[idx])
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
            data_[idx_, 6] = exec_time
            data_[idx_, 7] = deformable_im2col
            idx_ += 1
    df = pd.DataFrame(
        data_, columns=['BATCH', 'IN_CHANNELS', 'OUT_CHANNELS', 'IN_HEIGHT', 'IN_WIDTH', 'MODE', 'TOTAL_TIME(ms)', 'DEFORMABLE_IM2COL(ms)'])
    df['MODE'] = df['MODE'].map(lambda x: modes[int(x)])
    print(df)
    df.to_csv('results/deformable_run_times_%s_%s.csv' %
              (net_mode, time.time()), index=True)


if __name__ == '__main__':
    main()
