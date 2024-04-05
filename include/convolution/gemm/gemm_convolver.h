#ifndef __GEMM_CONVOLVER_H_
#define __GEMM_CONVOLVER_H_

#include <omp.h>

#include "../../log/logging.h"
#include "convolution/conv_base/convolver.h"
#include "core/mem/buffer.h"
#include "core/tensor/tensor.h"
#include "gemm/gemm.h"

#define NO_AVX 1

namespace conv2d {
template <typename T>
class GEMMConvolver : public Convolver<T> {
 private:
  GEMM<T> __gemm;
  std::shared_ptr<Buffer<T>> transformed_im_buffer;

 public:
  GEMMConvolver();
  void im2col(const Tensor<T> *filter, const Tensor<T> *input,
              Tensor<T> *output);
  void run(const Tensor<T> *filter, const Tensor<T> *input, Tensor<T> *output,
           stats_t *stats = nullptr);
  void gemm2im(const Tensor<T> *gemm_output, Tensor<T> *output);
  void set_switch(gemm_library lib_switch) { __gemm.set_switch(lib_switch); }
  ~GEMMConvolver(){};
};

template <typename T>
GEMMConvolver<T>::GEMMConvolver() {
  transformed_im_buffer = std::make_shared<Buffer<T>>(GetCPUAllocator<T>());
}

template <typename T>
void GEMMConvolver<T>::im2col(const Tensor<T> *filter, const Tensor<T> *input,
                              Tensor<T> *output) {
  const T *input_data = input->data();
  T *output_data = output->mutable_data();

  const index_t batches = input->dim(0);
  const index_t channels = input->dim(1);
  const index_t in_height = input->dim(2);
  const index_t in_width = input->dim(3);
  const index_t batch_size = channels * in_height * in_width;
  const index_t in_area = in_height * in_width;

  const index_t f_channels = filter->dim(1);
  const index_t filter_height = filter->dim(2);
  const index_t filter_width = filter->dim(3);
  const index_t filter_size = channels * filter_height * filter_width;
  const index_t filter_area = filter_height * filter_width;

  std::vector<index_t> *stride_ptr = this->get_stride();
  const index_t vertical_stride = stride_ptr->at(0);
  const index_t horizontal_stride = stride_ptr->at(1);

  if (f_channels != channels) {
    throw std::invalid_argument(
        "number of filters in the input and filter are not the same. input "
        "channels: " +
        std::to_string(channels) +
        " filter channels: " + std::to_string(f_channels));
  }

  const index_t output_height =
      this->compute_output_height(in_height, filter_height);
  const index_t output_width =
      this->compute_output_width(in_width, filter_width);
  const index_t out_area = output_height * output_width;
  const index_t transformed_out_area = out_area * filter_size;
  output->resize({batches, channels, out_area, filter_area});

  for (index_t b = 0; b < batches; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      for (index_t h = 0; h <= in_height - filter_height;
           h += vertical_stride) {
        for (index_t w = 0; w <= in_width - filter_width;
             w += horizontal_stride) {
          const T *input_batch_ptr = input_data + b * batch_size;
          T *out_batch_ptr = output_data + b * transformed_out_area;
          const T *input_channel_ptr = input_batch_ptr + c * in_area;
          T *out_channel_ptr = out_batch_ptr + c * filter_area;

          index_t out_idx =
              (h / vertical_stride) * output_width + (w / horizontal_stride);

          const T *input_ptr = input_channel_ptr + h * in_width + w;
          T *output_ptr = out_channel_ptr + out_idx * filter_size;
          for (index_t i = 0; i < filter_height; ++i) {
            memcpy(output_ptr + i * filter_width, input_ptr + i * in_width,
                   sizeof(T) * filter_width);
          }
        }
      }
    }
  }
}

template <typename T>
void GEMMConvolver<T>::gemm2im(const Tensor<T> *gemm_output,
                               Tensor<T> *output) {
  const T *gemm_out_data = gemm_output->data();
  T *out_data = output->mutable_data();

  const index_t batches = output->dim(0);
  const index_t num_filters = output->dim(1);
  const index_t out_height = output->dim(2);
  const index_t out_width = output->dim(3);
  const index_t out_area = out_height * out_width;
  const index_t batch_size = num_filters * out_area;

  for (index_t b = 0; b < batches; ++b) {
    for (index_t t = 0; t < out_area; ++t) {
      for (index_t j = 0; j < num_filters; ++j) {
        const T *gemm_batch_ptr = gemm_out_data + b * batch_size;
        const T *gemm_ptr = gemm_batch_ptr + t * num_filters;
        T *out_batch_ptr = out_data + b * batch_size;
        T *out_ptr = out_batch_ptr + t;
        out_ptr[j * out_area] = gemm_ptr[j];
      }
    }
  }
}

template <typename T>
void GEMMConvolver<T>::run(const Tensor<T> *filter, const Tensor<T> *input,
                           Tensor<T> *output, stats_t *stats) {
  const T *filter_data = filter->data();

  std::unique_ptr<Tensor<T>> padded_input = std::make_unique<Tensor<T>>();
  if (this->need_padding()) {
    this->pad_input(input, padded_input.get());
  }

  const index_t batches = input->dim(0);
  const index_t channels = input->dim(1);
  index_t in_height = input->dim(2);
  index_t in_width = input->dim(3);

  const index_t num_filters = filter->dim(0);
  const index_t f_channels = filter->dim(1);
  const index_t filter_height = filter->dim(2);
  const index_t filter_width = filter->dim(3);
  const index_t filter_area = filter_height * filter_width;
  const index_t filter_size = channels * filter_area;

  if (f_channels != channels) {
    throw std::invalid_argument(
        "number of filters in the input and filter are not the same. input "
        "channels: " +
        std::to_string(channels) +
        " filter channels: " + std::to_string(f_channels));
  }

  const index_t output_height =
      this->compute_output_height(in_height, filter_height);
  const index_t output_width =
      this->compute_output_width(in_width, filter_width);
  const index_t out_area = output_height * output_width;
  // const index_t out_batch_size = batches * num_filters * out_area;
  output->resize({batches, num_filters, output_height, output_width});

  /* create im2 col transformation */
  const index_t transformed_im_buffer_size =
      batches * channels * out_area * filter_area;
  // const index_t gemm_out_size = out_batch_size;
  const index_t bytes_required = transformed_im_buffer_size;
  transformed_im_buffer->resize(bytes_required);

  LOG_DEBUG("number of values to be written: " +
            std::to_string(transformed_im_buffer_size));
  transformed_im_buffer->clear();

  std::unique_ptr<Tensor<T>> transformed_im = std::make_unique<Tensor<T>>();
  transformed_im->resize({batches, channels, out_area, filter_area});
  T *transformed_im_data = transformed_im->mutable_data();

  std::unique_ptr<Tensor<T>> gemm_output = std::make_unique<Tensor<T>>();
  gemm_output->resize({1, 1, batches * out_area, num_filters});
  T *gemm_output_data = gemm_output->mutable_data();

  if (stats != nullptr) {
    stats->start_im2col = std::chrono::steady_clock::now();
  }
  if (this->need_padding()) {
    LOG_DEBUG("padding required for im2col");
    im2col(filter, padded_input.get(), transformed_im.get());
  } else {
    LOG_DEBUG("no padding required for im2col");
    im2col(filter, input, transformed_im.get());
  }
  if (stats != nullptr) {
    stats->end_im2col = std::chrono::steady_clock::now();
  }

  const T *A = transformed_im_data;
  const T *B = filter_data;
  T *C = gemm_output_data;
  const index_t m = batches * out_area;
  const index_t n = num_filters;
  const index_t k = filter_size;
  if (stats != nullptr) {
    stats->start_gemm = std::chrono::steady_clock::now();
  }
  __gemm.sgemm('N', 'T', m, n, k, 1, A, k, B, n, 1, C, n);
  gemm2im(gemm_output.get(), output);
  if (stats != nullptr) {
    stats->end_gemm = std::chrono::steady_clock::now();
  }
}

}  // namespace conv2d

#endif