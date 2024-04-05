#ifndef __WARP_DEFORM_CONV2D_H__
#define __WARP_DEFORM_CONV2D_H__

#include "cuda/dcn_deform.h"

at::Tensor warp_deform_conv2d(const at::Tensor& input, const at::Tensor& weight,
                              const at::Tensor& offset, const at::Tensor& mask,
                              const at::Tensor& bias, const int stride_h,
                              const int stride_w, const int pad_h,
                              const int pad_w, const int dilation_h,
                              const int dilation_w, const int groups,
                              const int offset_groups, const bool use_mask,
                              const float threshold) {
  return warp_deform_conv2d_forward_kernel(
      input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w,
      dilation_h, dilation_w, groups, offset_groups, use_mask, threshold);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_warp_deform_conv2d_backward(const at::Tensor& grad, const at::Tensor& input,
                             const at::Tensor& weight, const at::Tensor& offset,
                             const at::Tensor& mask, const at::Tensor& bias,
                             const int stride_h, const int stride_w,
                             const int pad_h, const int pad_w,
                             const int dilation_h, const int dilation_w,
                             const int groups, const int offset_groups,
                             const bool use_mask, const float threshold) {
  return _warp_deform_conv2d_backward_kernel(
      grad, input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w,
      dilation_h, dilation_w, groups, offset_groups, use_mask, threshold);
}

#endif