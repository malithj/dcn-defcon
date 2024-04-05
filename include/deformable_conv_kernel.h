#ifndef __DEFORMABLE_CONV_KERNEL_H__
#define __DEFORMABLE_CONV_KERNEL_H__

#include "types/types.h"

template <typename T>
void deformable_conv_kernel(T *offsets, T *input, index_t height, index_t width,
                            index_t kernel_height, index_t kernel_width,
                            index_t in_channels, index_t out_channels,
                            index_t batches);

#endif