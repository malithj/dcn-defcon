#include <math.h>

#include <algorithm>

#include "utils.h"

template <typename T>
T bilinear_interpolate(T *img, T *offsets, uint32_t height, uint32_t width, T x,
                       T y) {
  const int64_t height_ = static_cast<int64_t>(height);
  const int64_t width_ = static_cast<int64_t>(width);
  const int64_t low_x = static_cast<int64_t>(floor(x));  // x1
  const int64_t low_y = static_cast<int64_t>(floor(y));  // y1
  const int64_t high_x = low_x + 1;                      // x2
  const int64_t high_y = low_y + 1;                      // y2

  T img_high_x_high_y = 0;
  if (high_y < height_ && high_x < width_)
    img_high_x_high_y = img[high_y * width_ + high_x];
  T img_low_x_high_y = 0;
  if (high_y < height_ && low_x >= 0)
    img_low_x_high_y = img[high_y * width_ + low_x];
  T img_high_x_low_y = 0;
  if (low_y >= 0 && high_x < width_)
    img_high_x_low_y = img[low_y * width_ + high_x];
  T img_low_x_low_y = 0;
  if (low_y >= 0 && low_x >= 0) img_low_x_low_y = img[low_y * width_ + low_x];

  const float high_y_diff = high_y - y;  // y2 - y
  const float low_y_diff = y - low_y;    // y - y1
  const float high_x_diff = high_x - x;  // x2 - x
  const float low_x_diff = x - low_x;    // x - x1

  const float coeff_11 = high_y_diff * high_x_diff;
  const float coeff_12 = high_y_diff * low_x_diff;

  const float coeff_21 = low_y_diff * high_x_diff;
  const float coeff_22 = low_y_diff * low_x_diff;

  const float sum = coeff_11 * img_low_x_low_y + coeff_12 * img_high_x_low_y +
                    coeff_21 * img_low_x_high_y + coeff_22 * img_high_x_high_y;

  return sum;
}

// instatiation of the template
template float bilinear_interpolate<float>(float *img, float *offsets,
                                           uint32_t height, uint32_t width,
                                           float x, float y);