#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THC.h>
#include <cuda_fp16.h>

#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include "dcn_cuda_helpers.h"
#include "dcn_deform.h"

const int kMaxParallelImgs = 32;

#define CUDA_SAFE_CALL(err) __cudaSafeCall(err, __FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char* file, const int line) {
  if (cudaSuccess != err) {
    fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
    exit(-1);
  }
  return;
}

inline unsigned int GET_THREADS() {
#ifdef __HIP_PLATFORM_HCC__
  return 256;
#endif
  // thread count fixed to 512
  return 512;
  if (at::cuda::getCurrentDeviceProperties()->major >= 6) {
    return 1024;
  }
  return 512;
}

inline unsigned int GET_BLOCKS(const unsigned int THREADS,
                               const unsigned int N) {
  unsigned int kMaxGridNum =
      at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  return std::min(kMaxGridNum, (N + THREADS - 1) / THREADS);
}

__device__ float bilinear_interpolate(const float* in, int height, int width,
                                      float h, float w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0) v1 = in[h_low * width + w_low];
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1) v2 = in[h_low * width + w_high];
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0) v3 = in[h_high * width + w_low];
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = in[h_high * width + w_high];

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

__global__ void low_res_kernel(int n, const float* input, at::Half* output) {
  CUDA_1D_KERNEL_LOOP(index, n) { output[index] = __float2half(input[index]); }
}

__global__ void deformable_im2col_kernel(
    int n, const float* input_ptr, const float* offset_ptr,
    const float* mask_ptr, int height, int width, int weight_h, int weight_w,
    int pad_h, int pad_w, int stride_h, int stride_w, int dilation_h,
    int dilation_w, int batch_sz, int n_in_channels, int n_offset_grps,
    int out_h, int out_w, bool use_mask, const float threshold,
    float* columns_ptr) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int out_b = (index / (out_w * out_h)) % batch_sz;
    const int in_c = index / (out_w * out_h * batch_sz);
    const int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    const int grp_idx = in_c / c_per_offset_grp;

    columns_ptr += (out_c * (batch_sz * out_h * out_w) +
                    out_b * (out_h * out_w) + out_y * out_w + out_x);

    input_ptr +=
        (out_b * (n_in_channels * height * width) + in_c * (height * width));

    offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w *
                  out_h * out_w;

    mask_ptr += use_mask * ((out_b * n_offset_grps + grp_idx) * weight_h *
                            weight_w * out_h * out_w);

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const int mask_idx = i * weight_w + j;
        const int offset_idx = 2 * mask_idx;

        const float offset_h =
            offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const float offset_w = offset_ptr[(offset_idx + 1) * (out_h * out_w) +
                                          out_y * out_w + out_x];
        // add 0.5f for texture memory assignment
        // const float y = (out_y * stride_h - pad_h) + i * dilation_h +
        //                 offset_h * (offset_h <= threshold) + 0.5f;
        // const float x = (out_x * stride_w - pad_w) + j * dilation_w +
        //                 offset_w * (offset_w <= threshold) + 0.5f;
        // *columns_ptr =
        //     tex2DLayered<float>(tex_obj, x, y, out_b * n_in_channels + in_c);

        const float y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
        const float x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;
        *columns_ptr = bilinear_interpolate(input_ptr, height, width, y, x);
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}

__global__ void tex_deformable_im2col_kernel(
    int n, const float* offset_ptr, const float* mask_ptr, int height,
    int width, int weight_h, int weight_w, int pad_h, int pad_w, int stride_h,
    int stride_w, int dilation_h, int dilation_w, int batch_sz,
    int n_in_channels, int n_offset_grps, int out_h, int out_w, bool use_mask,
    const float threshold, float* columns_ptr, cudaTextureObject_t tex_obj) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int out_b = (index / (out_w * out_h)) % batch_sz;
    const int in_c = index / (out_w * out_h * batch_sz);
    const int out_c = in_c * weight_h * weight_w;

    int c_per_offset_grp = n_in_channels / n_offset_grps;
    const int grp_idx = in_c / c_per_offset_grp;

    columns_ptr += (out_c * (batch_sz * out_h * out_w) +
                    out_b * (out_h * out_w) + out_y * out_w + out_x);

    offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w *
                  out_h * out_w;

    mask_ptr += use_mask * ((out_b * n_offset_grps + grp_idx) * weight_h *
                            weight_w * out_h * out_w);

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const int mask_idx = i * weight_w + j;
        const int offset_idx = 2 * mask_idx;

        const float offset_h =
            offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const float offset_w = offset_ptr[(offset_idx + 1) * (out_h * out_w) +
                                          out_y * out_w + out_x];
        // add 0.5f for texture memory assignment
        const float y = (out_y * stride_h - pad_h) + i * dilation_h +
                        offset_h * (offset_h <= threshold) + 0.5f;
        const float x = (out_x * stride_w - pad_w) + j * dilation_w +
                        offset_w * (offset_w <= threshold) + 0.5f;
        *columns_ptr =
            tex2DLayered<float>(tex_obj, x, y, out_b * n_in_channels + in_c);
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }
}

void deformable_im2col(const at::Tensor& input, const at::Tensor& data_offset,
                       const at::Tensor& data_mask, int n_in_channels,
                       int height, int width, int weight_h, int weight_w,
                       int pad_h, int pad_w, int stride_h, int stride_w,
                       int dilation_h, int dilation_w, int out_h, int out_w,
                       int parallel_imgs, int deformable_group, bool use_mask,
                       const float threshold, at::Tensor data_col) {
  int num_kernels = n_in_channels * out_h * out_w * parallel_imgs;

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

  int* texDims = at::cuda::getCurrentDeviceProperties()->maxTexture2DLayered;
  if (height < texDims[0] && width < texDims[1] &&
      n_in_channels * parallel_imgs < texDims[2]) {
// Allocate CUDA array in device memory
// channel descriptor is the number of bits along x, y, z and w
#ifdef LOW_RES
    // low res image
    auto low_res_input =
        at::zeros({parallel_imgs, n_in_channels, height, width},
                  torch::TensorOptions()
                      .device(at::kCUDA, input.device().index())
                      .dtype(torch::kHalf));
    int num_in_kernels = parallel_imgs * n_in_channels * height * width;
    const unsigned int in_blocks = GET_BLOCKS(threads, num_in_kernels);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        low_res_input.scalar_type(), "low_res_kernel", ([&] {
          low_res_kernel<<<in_blocks, threads>>>(
              num_in_kernels, input.data_ptr<float>(),
              low_res_input.data_ptr<at::Half>());
        }));

    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
    // extent: width in bytes, height and depth [num layers]
    cudaExtent extent =
        make_cudaExtent(width, height, n_in_channels * parallel_imgs);
#else
    cudaChannelFormatDesc channelDesc =
        cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    // extent: width in bytes, height and depth [num layers]
    cudaExtent extent =
        make_cudaExtent(width, height, n_in_channels * parallel_imgs);
#endif
    cudaArray_t cu_3d_array;
    CUDA_SAFE_CALL(cudaMalloc3DArray(&cu_3d_array, &channelDesc, extent,
                                     cudaArrayLayered));

    // Set pitch of the source (the width in memory in bytes of the 2D array
    // pointed to by src, including padding), we dont have any padding
    //
    // The logical texture layout is as follows.
    //
    //           input height x input width
    //            |———————————————————|
    // in channels| . | . | . | . | . |
    //            | . | . | . | . | . |
    //            |———————————————————|
    // Copy data located at input tensor in host memory to device memory
    cudaMemcpy3DParms img_params = {0};
    img_params.srcPos = make_cudaPos(0, 0, 0);
    img_params.dstPos = make_cudaPos(0, 0, 0);
#ifdef LOW_RES
    img_params.srcPtr =
        make_cudaPitchedPtr(low_res_input.data_ptr<at::Half>(),
                            width * sizeof(at::Half), width, height);
#else
    img_params.srcPtr = make_cudaPitchedPtr(
        input.data_ptr<float>(), width * sizeof(float), width, height);
#endif
    img_params.dstArray = cu_3d_array;
    img_params.extent =
        make_cudaExtent(width, height, n_in_channels * parallel_imgs);
    img_params.kind = cudaMemcpyDeviceToDevice;
    CUDA_SAFE_CALL(cudaMemcpy3D(&img_params));

    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cu_3d_array;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    cudaTextureObject_t tex_obj = 0;
    CUDA_SAFE_CALL(cudaCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "tex_deformable_im2col", ([&] {
          tex_deformable_im2col_kernel<<<blocks, threads>>>(
              num_kernels, data_offset.data_ptr<float>(),
              data_mask.data_ptr<float>(), height, width, weight_h, weight_w,
              pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
              parallel_imgs, n_in_channels, deformable_group, out_h, out_w,
              use_mask, threshold, data_col.data_ptr<float>(), tex_obj);
        }));

    // Destroy texture object
    CUDA_SAFE_CALL(cudaDestroyTextureObject(tex_obj));
    CUDA_SAFE_CALL(cudaFreeArray(cu_3d_array));
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "deformable_im2col", ([&] {
          deformable_im2col_kernel<<<blocks, threads>>>(
              num_kernels, input.data_ptr<float>(),
              data_offset.data_ptr<float>(), data_mask.data_ptr<float>(),
              height, width, weight_h, weight_w, pad_h, pad_w, stride_h,
              stride_w, dilation_h, dilation_w, parallel_imgs, n_in_channels,
              deformable_group, out_h, out_w, use_mask, threshold,
              data_col.data_ptr<float>());
        }));
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
  }
}

int get_greatest_divisor_below_bound(int n, int bound) {
  for (int k = bound; k > 1; --k) {
    if (n % k == 0) {
      return k;
    }
  }
  return 1;
}

__global__ void deformable_col2im_kernel(
    int n, const float* col, const float* offset_ptr, const float* mask_ptr,
    int channels, int height, int width, int kernel_h, int kernel_w, int pad_h,
    int pad_w, int stride_h, int stride_w, int dilation_h, int dilation_w,
    int batch_sz, int n_offset_grps, int out_h, int out_w, bool use_mask,
    float* grad_im) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int out_x = index % out_w;
    const int out_y = (index / out_w) % out_h;
    const int b = (index / (out_w * out_h)) % batch_sz;
    const int j = (index / (out_w * out_h * batch_sz)) % kernel_w;
    const int i = (index / (out_w * out_h * batch_sz * kernel_w)) % kernel_h;
    const int c = index / (out_w * out_h * batch_sz * kernel_w * kernel_h);

    int c_per_offset_grp = channels / n_offset_grps;
    const int offset_grp = c / c_per_offset_grp;

    offset_ptr += (b * n_offset_grps + offset_grp) * 2 * kernel_h * kernel_w *
                  out_h * out_w;

    if (use_mask) {
      mask_ptr += (b * n_offset_grps + offset_grp) * kernel_h * kernel_w *
                  out_h * out_w;
    }

    const int mask_idx = i * kernel_w + j;
    const int offset_idx = 2 * mask_idx;

    const int offset_h_ptr = ((offset_idx)*out_h + out_y) * out_w + out_x;
    const int offset_w_ptr = ((offset_idx + 1) * out_h + out_y) * out_w + out_x;

    const float offset_h = offset_ptr[offset_h_ptr];
    const float offset_w = offset_ptr[offset_w_ptr];

    float mask_value = 1;
    if (use_mask) {
      mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x];
    }

    const float y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
    const float x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int yp = int(y) + dy;
        int xp = int(x) + dx;
        if (0 <= yp && yp < height && 0 <= xp && xp < width &&
            std::abs(y - yp) < 1 && std::abs(x - xp) < 1) {
          int grad_pos = ((b * channels + c) * height + yp) * width + xp;
          float weight = (1 - std::abs(y - yp)) * (1 - std::abs(x - xp));
          atomicAdd(grad_im + grad_pos, mask_value * weight * col[index]);
        }
      }
    }
  }
}

void compute_grad_input(const at::Tensor& columns, const at::Tensor& offset,
                        const at::Tensor& mask, int channels, int height,
                        int width, int weight_h, int weight_w, int pad_h,
                        int pad_w, int stride_h, int stride_w, int dilation_h,
                        int dilation_w, int parallel_imgs, int n_offset_grps,
                        bool use_mask, at::Tensor grad_im) {
  int out_h =
      (height + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
  int out_w =
      (width + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
  int num_kernels =
      channels * weight_h * weight_w * out_h * out_w * parallel_imgs;

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      columns.scalar_type(), "compute_grad_input", ([&] {
        deformable_col2im_kernel<<<blocks, threads>>>(
            num_kernels, columns.data_ptr<float>(), offset.data_ptr<float>(),
            mask.data_ptr<float>(), channels, height, width, weight_h, weight_w,
            pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w,
            parallel_imgs, n_offset_grps, out_h, out_w, use_mask,
            grad_im.data_ptr<float>());
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in compute_grad_input: %s\n", cudaGetErrorString(err));
  }
}

__device__ float get_coordinate_weight(const float* im_data, int height,
                                       int width, float y, float x,
                                       bool is_y_direction) {
  int y_l = floor(y);
  int x_l = floor(x);
  int y_h = y_l + 1;
  int x_h = x_l + 1;

  bool valid_y_l = 0 <= y_l && y_l < height;
  bool valid_y_h = 0 <= y_h && y_h < height;
  bool valid_x_l = 0 <= x_l && x_l < width;
  bool valid_x_h = 0 <= x_h && x_h < width;

  float zero = 0;
  float v_yx = (valid_y_l && valid_x_l) ? im_data[y_l * width + x_l] : zero;
  float v_yX = (valid_y_l && valid_x_h) ? im_data[y_l * width + x_h] : zero;
  float v_Yx = (valid_y_h && valid_x_l) ? im_data[y_h * width + x_l] : zero;
  float v_YX = (valid_y_h && valid_x_h) ? im_data[y_h * width + x_h] : zero;

  if (is_y_direction) {
    float dx = x - x_l;
    return dx * (v_YX - v_yX) + (1 - dx) * (v_Yx - v_yx);
  } else {
    float dy = y - y_l;
    return dy * (v_YX - v_Yx) + (1 - dy) * (v_yX - v_yx);
  }
}

__global__ void deformable_col2im_coord_kernel(
    int n, const float* col_ptr, const float* im_ptr, const float* offset_ptr,
    const float* mask_ptr, int channels, int height, int width, int weight_h,
    int weight_w, int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int batch_sz, int offset_channels,
    int n_offset_grps, int out_h, int out_w, const bool use_mask,
    float* grad_offset, float* grad_mask) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    float grad_offset_val = 0;
    float grad_mask_val = 0;

    int w = index % out_w;
    int h = (index / out_w) % out_h;
    int w_w = (index / (out_w * out_h * 2)) % weight_w;
    int w_h = (index / (out_w * out_h * 2 * weight_w)) % weight_h;
    int c = (index / (out_w * out_h)) % offset_channels;
    int b = index / (out_w * out_h * offset_channels);

    const int offset_grp = c / (2 * weight_h * weight_w);
    const int col_step = weight_h * weight_w;

    int c_per_offset_grp = channels / n_offset_grps;

    col_ptr += offset_grp * c_per_offset_grp * weight_h * weight_w * batch_sz *
               out_w * out_h;
    im_ptr +=
        (b * n_offset_grps + offset_grp) * c_per_offset_grp * height * width;
    offset_ptr += (b * n_offset_grps + offset_grp) * 2 * weight_h * weight_w *
                  out_h * out_w;

    if (use_mask) {
      mask_ptr += (b * n_offset_grps + offset_grp) * weight_h * weight_w *
                  out_h * out_w;
    }

    const int offset_c = c - offset_grp * 2 * weight_h * weight_w;
    const bool is_y_direction = offset_c % 2 == 0;

    const int c_bound = c_per_offset_grp * weight_h * weight_w;
    for (int col_c = (offset_c / 2); col_c < c_bound; col_c += col_step) {
      const int col_pos = (((col_c * batch_sz + b) * out_h) + h) * out_w + w;

      int out_x = col_pos % out_w;
      int out_y = (col_pos / out_w) % out_h;
      int j = (col_pos / (out_w * out_h * batch_sz)) % weight_w;
      int i = (col_pos / (out_w * out_h * batch_sz * weight_w)) % weight_h;

      const int mask_idx = i * weight_w + j;

      const int offset_h_ptr =
          (((2 * mask_idx) * out_h + out_y) * out_w + out_x);
      const int offset_w_ptr =
          (((2 * mask_idx + 1) * out_h + out_y) * out_w + out_x);
      const float offset_h = offset_ptr[offset_h_ptr];
      const float offset_w = offset_ptr[offset_w_ptr];

      float mask_value = 1;
      if (use_mask) {
        mask_value = mask_ptr[(mask_idx * out_h + out_y) * out_w + out_x];
      }

      float y = (out_y * stride_h - pad_h) + i * dilation_h + offset_h;
      float x = (out_x * stride_w - pad_w) + j * dilation_w + offset_w;

      const float weight =
          get_coordinate_weight(im_ptr, height, width, y, x, is_y_direction);
      grad_offset_val += mask_value * weight * col_ptr[col_pos];

      if (use_mask && is_y_direction) {
        grad_mask_val += col_ptr[col_pos] *
                         bilinear_interpolate(im_ptr, height, width, y, x);
      }

      im_ptr += height * width;
    }

    grad_offset[index] = grad_offset_val;

    if (use_mask && is_y_direction) {
      const int idx =
          ((((b * n_offset_grps + offset_grp) * weight_h + w_h) * weight_w +
            w_w) *
               out_h +
           h) *
              out_w +
          w;
      grad_mask[idx] = grad_mask_val;
    }
  }
}

void compute_grad_offset_and_mask(
    const at::Tensor& columns, const at::Tensor& input,
    const at::Tensor& offset, const at::Tensor& mask, int channels, int height,
    int width, int weight_h, int weight_w, int pad_h, int pad_w, int stride_h,
    int stride_w, int dilation_h, int dilation_w, int parallel_imgs,
    int n_offset_grps, bool use_mask, at::Tensor grad_offset,
    at::Tensor grad_mask) {
  int out_h =
      (height + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;
  int out_w =
      (width + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
  int num_kernels =
      out_h * out_w * 2 * weight_h * weight_w * n_offset_grps * parallel_imgs;

  const unsigned int threads = GET_THREADS();
  const unsigned int blocks = GET_BLOCKS(threads, num_kernels);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      columns.scalar_type(), "compute_grad_offset_and_mask", ([&] {
        deformable_col2im_coord_kernel<<<blocks, threads>>>(
            num_kernels, columns.data_ptr<float>(), input.data_ptr<float>(),
            offset.data_ptr<float>(), mask.data_ptr<float>(), channels, height,
            width, weight_h, weight_w, pad_h, pad_w, stride_h, stride_w,
            dilation_h, dilation_w, parallel_imgs,
            2 * weight_h * weight_w * n_offset_grps, n_offset_grps, out_h,
            out_w, use_mask, grad_offset.data_ptr<float>(),
            grad_mask.data_ptr<float>());
      }));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in compute_grad_offset_and_mask: %s\n",
           cudaGetErrorString(err));
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> backward_gradient_inputs(
    at::Tensor input, at::Tensor weight, at::Tensor offset, at::Tensor mask,
    at::Tensor grad_out, int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation_h, int dilation_w, int n_weight_grps, int n_offset_grps,
    int n_parallel_imgs, bool use_mask) {
  at::DeviceGuard guard(input.device());

  int batch_sz = input.size(0);
  long n_in_channels = input.size(1);
  long in_h = input.size(2);
  long in_w = input.size(3);

  n_parallel_imgs = std::min(batch_sz, n_parallel_imgs);

  long n_out_channels = weight.size(0);
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  long out_w =
      (in_w + 2 * pad_w - (dilation_w * (weight_w - 1) + 1)) / stride_w + 1;
  long out_h =
      (in_h + 2 * pad_h - (dilation_h * (weight_h - 1) + 1)) / stride_h + 1;

  auto grad_input = at::zeros_like(input);
  auto grad_offset = at::zeros_like(offset);
  auto grad_mask = at::zeros_like(mask);

  if (batch_sz == 0) {
    return std::make_tuple(grad_input, grad_offset, grad_mask);
  }

  auto columns = at::empty(
      {n_in_channels * weight_w * weight_h, n_parallel_imgs * out_h * out_w},
      input.options());

  // Separate into blocks
  grad_input = grad_input.reshape(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});
  input = input.reshape(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});

  grad_offset = grad_offset.reshape(
      {batch_sz / n_parallel_imgs, n_parallel_imgs,
       n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});
  offset =
      offset.reshape({batch_sz / n_parallel_imgs, n_parallel_imgs,
                      n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

  if (use_mask) {
    grad_mask =
        grad_mask.reshape({batch_sz / n_parallel_imgs, n_parallel_imgs,
                           n_offset_grps * weight_h * weight_w, out_h, out_w});
    mask = mask.reshape({batch_sz / n_parallel_imgs, n_parallel_imgs,
                         n_offset_grps * weight_h * weight_w, out_h, out_w});
  }

  grad_out =
      grad_out
          .reshape({batch_sz / n_parallel_imgs, n_parallel_imgs, n_weight_grps,
                    n_out_channels / n_weight_grps, out_h, out_w})
          .permute({0, 2, 3, 1, 4, 5});

  weight = weight.reshape({n_weight_grps, weight.size(0) / n_weight_grps,
                           weight.size(1), weight.size(2), weight.size(3)});

  columns = columns.view(
      {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
  for (int elt = 0; elt < batch_sz / n_parallel_imgs; elt++) {
    columns.zero_();
    // Separate into weight groups
    for (int g = 0; g < n_weight_grps; g++) {
      columns[g] = columns[g].addmm_(weight[g].flatten(1).transpose(0, 1),
                                     grad_out[elt][g].flatten(1));
    }

    compute_grad_offset_and_mask(columns, input[elt], offset[elt], mask[elt],
                                 n_in_channels, in_h, in_w, weight_h, weight_w,
                                 pad_h, pad_w, stride_h, stride_w, dilation_h,
                                 dilation_w, n_parallel_imgs, n_offset_grps,
                                 use_mask, grad_offset[elt], grad_mask[elt]);

    compute_grad_input(columns, offset[elt], mask[elt], n_in_channels, in_h,
                       in_w, weight_h, weight_w, pad_h, pad_w, stride_h,
                       stride_w, dilation_h, dilation_w, n_parallel_imgs,
                       n_offset_grps, use_mask, grad_input[elt]);
  }

  grad_input = grad_input.view({batch_sz, n_in_channels, in_h, in_w});
  grad_offset = grad_offset.view(
      {batch_sz, n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

  if (use_mask) {
    grad_mask = grad_mask.view(
        {batch_sz, n_offset_grps * weight_h * weight_w, out_h, out_w});
  }

  return std::make_tuple(grad_input, grad_offset, grad_mask);
}

at::Tensor backward_gradient_parameters(
    at::Tensor input, const at::Tensor& weight, at::Tensor offset,
    at::Tensor mask, const at::Tensor& grad_out, int stride_h, int stride_w,
    int pad_h, int pad_w, int dilation_h, int dilation_w, int n_weight_grps,
    int n_offset_grps, int n_parallel_imgs, bool use_mask,
    const float threshold) {
  at::DeviceGuard guard(input.device());

  int batch_sz = input.size(0);
  long n_in_channels = input.size(1);
  long in_h = input.size(2);
  long in_w = input.size(3);

  n_parallel_imgs = std::min(batch_sz, n_parallel_imgs);

  long n_out_channels = weight.size(0);
  int weight_h = weight.size(2);
  int weight_w = weight.size(3);

  long out_h = grad_out.size(2);
  long out_w = grad_out.size(3);

  auto grad_weight = at::zeros_like(weight);
  if (batch_sz == 0) {
    return grad_weight;
  }

  at::Tensor grad_out_buf =
      grad_out
          .reshape({batch_sz / n_parallel_imgs, n_parallel_imgs, n_weight_grps,
                    n_out_channels / n_weight_grps, out_h, out_w})
          .permute({0, 2, 3, 1, 4, 5})
          .contiguous();

  input = input.reshape(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, n_in_channels, in_h, in_w});

  offset =
      offset.reshape({batch_sz / n_parallel_imgs, n_parallel_imgs,
                      n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

  if (use_mask) {
    mask = mask.reshape({batch_sz / n_parallel_imgs, n_parallel_imgs,
                         n_offset_grps * weight_h * weight_w, out_h, out_w});
  }

  grad_weight = grad_weight.reshape(
      {n_weight_grps, grad_weight.size(0) / n_weight_grps, grad_weight.size(1),
       grad_weight.size(2), grad_weight.size(3)});

  auto columns = at::empty(
      {n_weight_grps, n_in_channels * weight_w * weight_h / n_weight_grps,
       n_parallel_imgs * out_h * out_w},
      input.options());

  for (int elt = 0; elt < batch_sz / n_parallel_imgs; elt++) {
    deformable_im2col(input[elt], offset[elt], mask[elt], n_in_channels, in_h,
                      in_w, weight_h, weight_w, pad_h, pad_w, stride_h,
                      stride_w, dilation_h, dilation_w, out_h, out_w,
                      n_parallel_imgs, n_offset_grps, use_mask, threshold,
                      columns);

    for (int g = 0; g < n_weight_grps; g++) {
      grad_weight[g] = grad_weight[g]
                           .flatten(1)
                           .addmm_(grad_out_buf[elt][g].flatten(1),
                                   columns[g].transpose(1, 0))
                           .view_as(grad_weight[g]);
    }
  }

  grad_weight = grad_weight.view({grad_weight.size(0) * grad_weight.size(1),
                                  grad_weight.size(2), grad_weight.size(3),
                                  grad_weight.size(4)});
  return grad_weight;
}

at::Tensor warp_deform_conv2d_forward_kernel(
    const at::Tensor& input, const at::Tensor& weight, const at::Tensor& offset,
    const at::Tensor& mask, const at::Tensor& bias, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int n_weight_grps, const int n_offset_grps,
    bool use_mask, const float threshold) {
  at::Tensor input_c = input.contiguous();
  at::Tensor offset_c = offset.contiguous();
  at::Tensor weight_c = weight.contiguous();
  at::Tensor mask_c = mask.contiguous();
  at::Tensor bias_c = bias.contiguous();

  TORCH_CHECK(input_c.ndimension() == 4);
  TORCH_CHECK(offset_c.ndimension() == 4);
  TORCH_CHECK(!use_mask || mask_c.ndimension() == 4);
  TORCH_CHECK(weight_c.ndimension() == 4);
  TORCH_CHECK(input_c.is_cuda(), "input must be a CUDA tensor");

  at::DeviceGuard guard(input_c.device());

  int batch_sz = input_c.size(0);
  int in_channels = input_c.size(1);
  int in_h = input_c.size(2);
  int in_w = input_c.size(3);

  int n_parallel_imgs =
      get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

  int out_channels = weight_c.size(0);
  int weight_h = weight_c.size(2);
  int weight_w = weight_c.size(3);

  int ker_h = dilation_h * (weight_h - 1) + 1;
  int ker_w = dilation_w * (weight_w - 1) + 1;
  int out_h = ((in_h + 2 * pad_h - ker_h) / stride_h) + 1;
  int out_w = ((in_w + 2 * pad_w - ker_w) / stride_w) + 1;

  TORCH_CHECK(weight_h > 0 && weight_w > 0, "weight_h: ", weight_h,
              " weight_w: ", weight_w);
  TORCH_CHECK(stride_h > 0 && stride_w > 0, "stride_h: ", stride_h,
              " stride_w: ", stride_w);
  TORCH_CHECK(pad_h >= 0 && pad_w >= 0, "pad_h: ", pad_h, " pad_w: ", pad_w);
  TORCH_CHECK(dilation_h > 0 && dilation_w > 0, "dilation_h: ", dilation_h,
              " dilation_w: ", dilation_w);

  TORCH_CHECK(weight_c.size(1) * n_weight_grps == input_c.size(1));
  TORCH_CHECK(weight_c.size(0) % n_weight_grps == 0);
  TORCH_CHECK((offset_c.size(1) == n_offset_grps * 2 * weight_h * weight_w),
              "offset.shape[1] is not valid: got: ", offset_c.size(1),
              " expected: ", n_offset_grps * 2 * weight_h * weight_w);
  TORCH_CHECK(
      (!use_mask || mask_c.size(1) == n_offset_grps * weight_h * weight_w),
      "mask.shape[1] is not valid: got: ", mask_c.size(1),
      " expected: ", n_offset_grps * weight_h * weight_w);
  TORCH_CHECK(input_c.size(1) % n_offset_grps == 0);

  TORCH_CHECK((offset_c.size(0) == input_c.size(0)),
              "invalid batch size of offset");
  TORCH_CHECK((offset_c.size(2) == out_h && offset_c.size(3) == out_w),
              "offset output dims: (", offset_c.size(2), ", ", offset_c.size(3),
              ") - ", "computed output dims: (", out_h, ", ", out_w, ")");
  TORCH_CHECK((mask_c.size(0) == input_c.size(0)),
              "invalid batch size of mask");
  TORCH_CHECK(
      (!use_mask || (mask_c.size(2) == out_h && mask_c.size(3) == out_w)),
      "mask output dims: (", mask_c.size(2), ", ", mask_c.size(3), ") - ",
      "computed output dims: (", out_h, ", ", out_w, ")");
  TORCH_CHECK(out_h > 0 && out_w > 0,
              "Calculated output size too small - out_h: ", out_h,
              " out_w: ", out_w);

  auto out =
      at::zeros({batch_sz, out_channels, out_h, out_w}, input_c.options());
  if (batch_sz == 0) {
    return out;
  }

  // Separate batches into blocks
  out = out.view({batch_sz / n_parallel_imgs, n_parallel_imgs, out_channels,
                  out_h, out_w});
  input_c = input_c.view(
      {batch_sz / n_parallel_imgs, n_parallel_imgs, in_channels, in_h, in_w});

  offset_c =
      offset_c.view({batch_sz / n_parallel_imgs, n_parallel_imgs,
                     n_offset_grps * 2 * weight_h * weight_w, out_h, out_w});

  if (use_mask) {
    mask_c = mask_c.view({batch_sz / n_parallel_imgs, n_parallel_imgs,
                          n_offset_grps * weight_h * weight_w, out_h, out_w});
  }

  at::Tensor out_buf = at::zeros({batch_sz / n_parallel_imgs, out_channels,
                                  n_parallel_imgs * out_h, out_w},
                                 out.options());

  // Separate channels into convolution groups
  out_buf = out_buf.view({out_buf.size(0), n_weight_grps,
                          out_buf.size(1) / n_weight_grps, out_buf.size(2),
                          out_buf.size(3)});
  weight_c =
      weight_c.view({n_weight_grps, weight_c.size(0) / n_weight_grps,
                     weight_c.size(1), weight_c.size(2), weight_c.size(3)});

  // Sample points and perform convolution
  auto columns = at::zeros(
      {in_channels * weight_h * weight_w, n_parallel_imgs * out_h * out_w},
      input_c.options());
  for (int b = 0; b < batch_sz / n_parallel_imgs; b++) {
    deformable_im2col(input_c[b], offset_c[b], mask_c[b], in_channels, in_h,
                      in_w, weight_h, weight_w, pad_h, pad_w, stride_h,
                      stride_w, dilation_h, dilation_w, out_h, out_w,
                      n_parallel_imgs, n_offset_grps, use_mask, threshold,
                      columns);

    columns = columns.view(
        {n_weight_grps, columns.size(0) / n_weight_grps, columns.size(1)});
    for (int g = 0; g < n_weight_grps; g++) {
      out_buf[b][g] = out_buf[b][g]
                          .flatten(1)
                          .addmm_(weight_c[g].flatten(1), columns[g])
                          .view_as(out_buf[b][g]);
    }
    columns =
        columns.view({columns.size(0) * columns.size(1), columns.size(2)});
  }

  out_buf = out_buf.view({batch_sz / n_parallel_imgs, out_channels,
                          n_parallel_imgs, out_h, out_w});
  out_buf.transpose_(1, 2);
  out.copy_(out_buf);
  out = out.view({batch_sz, out_channels, out_h, out_w});

  return out + bias_c.view({1, out_channels, 1, 1});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
_warp_deform_conv2d_backward_kernel(
    const at::Tensor& grad_out, const at::Tensor& input,
    const at::Tensor& weight, const at::Tensor& offset, const at::Tensor& mask,
    const at::Tensor& bias, const int stride_h, const int stride_w,
    const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int n_weight_grps, const int n_offset_grps,
    bool use_mask, const float threshold) {
  at::Tensor grad_out_c = grad_out.contiguous();
  at::Tensor input_c = input.contiguous();
  at::Tensor weight_c = weight.contiguous();
  at::Tensor offset_c = offset.contiguous();
  at::Tensor mask_c = mask.contiguous();
  at::Tensor bias_c = bias.contiguous();

  const int batch_sz = input_c.size(0);
  const int n_parallel_imgs =
      get_greatest_divisor_below_bound(batch_sz, kMaxParallelImgs);

  auto grad_input_and_offset_and_mask = backward_gradient_inputs(
      input_c, weight_c, offset_c, mask_c, grad_out_c, stride_h, stride_w,
      pad_h, pad_w, dilation_h, dilation_w, n_weight_grps, n_offset_grps,
      n_parallel_imgs, use_mask);

  auto grad_input = std::get<0>(grad_input_and_offset_and_mask);
  auto grad_offset = std::get<1>(grad_input_and_offset_and_mask);
  auto grad_mask = std::get<2>(grad_input_and_offset_and_mask);

  auto grad_weight = backward_gradient_parameters(
      input_c, weight_c, offset_c, mask_c, grad_out_c, stride_h, stride_w,
      pad_h, pad_w, dilation_h, dilation_w, n_weight_grps, n_offset_grps,
      n_parallel_imgs, use_mask, threshold);

  auto value = grad_out_c.sum({0, 2, 3});
  auto grad_bias = at::ones_like(bias_c) * value;

  return std::make_tuple(grad_input, grad_weight, grad_offset, grad_mask,
                         grad_bias);
}
