#include "warp_deform_conv2d.h"

#include <pybind11/pybind11.h>

#include "torch_warp_deform_conv2d.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("warp_dcn_v3_forward", &warp_deform_conv2d, "warp_dcn_v3_forward");
  m.def("_warp_dcn_v3_backward", &_warp_deform_conv2d_backward,
        "_warp_dcn_v3_backward");
  m.def("torch_warp_dcn_v3_forward", &torch_warp_deform_conv2d,
        "torch_warp_dcn_v3_forward");
  m.def("_torch_warp_dcn_v3_backward", &_torch_warp_deform_conv2d_backward,
        "_torch_warp_dcn_v3_backward");
}