from itertools import islice
import os
import time

import numpy as np
import pandas as pd


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
    count = len(batches) * len(in_channels) * len(out_channels) * len(modes)
    prog_iterations = 100
    global_df = pd.DataFrame()
    run = False
    flags = 'tex_cache_hit_rate,tex_cache_throughput,tex_cache_transactions,tex_utilization,texture_load_requests,l2_tex_read_hit_rate,l2_tex_read_throughput,l2_read_throughput,l1_cache_global_hit_rate,l2_l1_read_hit_rate,l2_l1_read_throughput,stall_texture,ldst_executed,ldst_issued,tex_fu_utilization,inst_executed_tex_ops,dram_read_bytes,flop_count_sp,gld_transactions_per_request,gld_efficiency,gld_transactions'
    net_mode = 'full'

    for lidx, layer in enumerate(layers):
        b = batches[0]
        c = layer[0]
        f = layer[1]
        in_height = layer[2]
        in_width = layer[3]
        for m_idx, m in enumerate(modes):
            template_command = "/usr/local/cuda/bin/nvprof --unified-memory-profiling off --log-file 'results/%s.csv' --csv --metrics %s python exported_model/run_model.py --mode %s --in_channels %s --out_channels %s --in_height %s --in_width %s --batch %s --itr %s --net_mode %s --layer %s"
            command = template_command % (
                'stats_%s_%s' % (str(c), m), flags, m, c, f, in_height, in_width, b, prog_iterations, net_mode, layer_idx_lookup[lidx])
            print("running command: ", command)
            ret = os.system(command)
            try:
                # os.system("sed -i -e 1,4d results/stats.csv")
                f_ = open("results/stats_%s_%s.csv" % (str(c), m), "r")
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
                try:
                    file_name = "results/tmp_stats_%s_%s.csv" % (
                        str(c), m)
                    df = pd.read_csv(file_name, delimiter=',',
                                     skipinitialspace=True)
                except Exception as process_e:
                    print("error while processing data", process_e)
            except Exception as e:
                print(
                    "error detected while removing comments on profiled file", e)
                continue
            df['MODE'] = m
            df['BATCH'] = b
            df['IN_CHANNEL'] = c
            df['OUT_CHANNEL'] = f
            df['IN_HEIGHT'] = in_height
            df['IN_WIDTH'] = in_width
            if not run:
                global_df = df
                run = True
            else:
                global_df = global_df.append(df)
    global_df.to_csv('results/profiled_deformable_run_times_%s.csv' %
                     (time.time()), index=False)


if __name__ == '__main__':
    main()
