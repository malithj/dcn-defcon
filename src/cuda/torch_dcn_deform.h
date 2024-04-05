#ifndef __TORCH_DCN_DEFORM_H__
#define __TORCH_DCN_DEFORM_H__

#include <torch/extension.h>

namespace torchdcn {
at::Tensor torch_warp_deform_conv2d_forward_kernel(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& offset,
    const at::Tensor& mask, const at::Tensor& bias, int64_t stride_h,
    int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dilation_h,
    int64_t dilation_w, int64_t n_weight_grps, int64_t n_offset_grps,
    bool use_mask);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_torch_warp_deform_conv2d_backward_kernel(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& offset, const at::Tensor& mask,
    const at::Tensor& bias, int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w, int64_t dilation_h,
    int64_t dilation_w, int64_t n_weight_grps, int64_t n_offset_grps,
    bool use_mask);
}  // namespace torchdcn

#endif
