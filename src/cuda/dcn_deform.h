#ifndef __DCN_DEFORM_H__
#define __DCN_DEFORM_H__

#include <torch/extension.h>

at::Tensor warp_deform_conv2d_forward_kernel(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& offset,
    const at::Tensor& mask, const at::Tensor& bias, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int n_weight_grps, const int n_offset_grps,
    bool use_mask, const float threshold);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_warp_deform_conv2d_backward_kernel(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& offset, const at::Tensor& mask,
    const at::Tensor& bias, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int n_weight_grps, const int n_offset_grps,
    bool use_mask, const float threshold);

#endif