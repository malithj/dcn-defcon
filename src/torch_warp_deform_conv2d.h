#ifndef __TORCH_WARP_DEFORM_CONV2D_H__
#define __TORCH_WARP_DEFORM_CONV2D_H__

#include "cuda/torch_dcn_deform.h"

using namespace torchdcn;

at::Tensor torch_warp_deform_conv2d(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& offset,
    const at::Tensor& mask, const at::Tensor& bias, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int groups, const int offset_groups,
    const bool use_mask) {
  return torch_warp_deform_conv2d_forward_kernel(
      input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w,
      dilation_h, dilation_w, groups, offset_groups, use_mask);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_torch_warp_deform_conv2d_backward(
    const at::Tensor& grad, const at::Tensor& input, const at::Tensor& weight,
    const at::Tensor& offset, const at::Tensor& mask, const at::Tensor& bias,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, const int groups,
    const int offset_groups, const bool use_mask) {
  return _torch_warp_deform_conv2d_backward_kernel(
      grad, input, weight, offset, mask, bias, stride_h, stride_w, pad_h, pad_w,
      dilation_h, dilation_w, groups, offset_groups, use_mask);
}

#endif