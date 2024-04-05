#ifndef __TYPES_H__
#define __TYPES_H__

#include <chrono>
#include <cstddef>
#include <cstdint>

typedef uint64_t index_t;
typedef enum {
  WINO_K_3x3,
  WINO_K_3x4,
  WINO_K_4x4,
  WINO_K_4x3,
  WINO_K_5x5,
  WINO_K_7x7
} wino_k_t;
typedef enum {
  WINO_O_2x2,
  WINO_O_3x3,
  WINO_O_4x4,
  WINO_O_5x5,
  WINO_O_6x6,
  WINO_O_7x7,
  WINO_O_8x8
} wino_o_t;
typedef enum { MKL, ONEDNN } gemm_library;
struct stats_t {
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point start_input_transform;
  std::chrono::steady_clock::time_point end_input_transform;
  std::chrono::steady_clock::time_point start_filter_transform;
  std::chrono::steady_clock::time_point end_filter_transform;
  std::chrono::steady_clock::time_point start_output_transform;
  std::chrono::steady_clock::time_point end_output_transform;
  std::chrono::steady_clock::time_point start_hardamard_transform;
  std::chrono::steady_clock::time_point end_hardamard_transform;
  std::chrono::steady_clock::time_point start_gemm;
  std::chrono::steady_clock::time_point end_gemm;
  std::chrono::steady_clock::time_point start_im2col;
  std::chrono::steady_clock::time_point end_im2col;
};

#endif