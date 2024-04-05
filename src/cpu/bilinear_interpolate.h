#ifndef __BILINEAR_INTERPLOATE_H__
#define __BILINEAR_INTERPLOATE_H___

#include <types/types.h>

template <typename T>
T bilinear_interpolate(T *img, T *offsets, uint32_t height, uint32_t width, T x,
                       T y);

#endif