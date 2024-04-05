#ifndef __SGEMM_H_
#define __SGEMM_H_

#include <string.h>

#include <memory>

#include "../log/logging.h"
#include "../types/types.h"

#define NO_NEON 1

template <typename T>
class SGEMM {
 private:
  std::unique_ptr<Logging::LoggingInternals::Logger> logger;
  const index_t N = 64;

 public:
  SGEMM<T>();
  // Performs GEMM (General Matrix Multiplication)
  // block size for all three dimensions is set to the above "N" value by
  // default.
  // The following variables are used to define the functionality (compatible
  // with available BLAS APIs)
  //
  // parameters
  // ----------
  // transa - whether to transpose A ('N|n': use original, 'T|t': transpose)
  // transb - whether to transpose B ('N|n': use original, 'T|t': transpose)
  // m      - number of rows in matrix A (row major)
  // n      - number of columns in matrix B (row major)
  // k      - numer of columns in matrix A and rows in matrix B (row major)
  // alpha  - factor of matrix A (alpha * AB + beta * C) @TODO(malith):
  // a      - pointer of type T of matrix A
  // lda    - leading dimension of matrix A (stride offset) @TODO(malith):
  // b      - pointer of type T of matrix B
  // ldb    - leading dimension of matrix B (stride offset) @TODO(malith):
  // beta   - factor of matrix B (alpha * AB + beta * C) @TODO(malith):
  // c      - pointer of type T of matrix C
  // ldc    - leading dimension of matrix C (stride offset) @TODO(malith):
  index_t gemm(char transa, char transb, index_t m, index_t n, index_t k,
               T alpha, const T *a, index_t lda, const T *b, index_t ldb,
               T beta, T *c, index_t ldc);
  ~SGEMM<T>(){};
};

template <typename T>
SGEMM<T>::SGEMM() {
  this->logger = std::make_unique<Logging::LoggingInternals::Logger>(__FILE__);
}

template <typename T>
index_t SGEMM<T>::gemm(char transa, char transb, index_t m, index_t n,
                       index_t k, T alpha, const T *a, index_t lda, const T *b,
                       index_t ldb, T beta, T *c, index_t ldc) {
  index_t a_t = 0;
  index_t b_t = 0;
  if (transa == 'N' || transa == 'n') {
    LOG_DEBUG("matrix A is not being transposed");
  } else if (transa == 'T' || transa == 'T') {
    LOG_DEBUG("matrix A is being transposed");
    a_t = 1;
  } else {
    std::string mode(1, transa);
    throw std::runtime_error("unknown matrix mode provided: " + mode);
  }
  if (transb == 'N' || transb == 'n') {
    LOG_DEBUG("matrix B is not being transposed");
  } else if (transb == 'T' || transb == 'T') {
    LOG_DEBUG("matrix B is being transposed");
    b_t = 1;
  } else {
    std::string mode(1, transb);
    throw std::runtime_error("unknown matrix mode provided: " + mode);
  }

  const index_t i_block_size = std::min(m, N);
  const index_t j_block_size = std::min(n, N);
  const index_t k_block_size = std::min(k, N);

  LOG_DEBUG("block size for i idx chosen as: " + std::to_string(i_block_size));
  LOG_DEBUG("block size for j idx chosen as: " + std::to_string(j_block_size));
  LOG_DEBUG("block size for k idx chosen as: " + std::to_string(k_block_size));

  // init c matrix to zero
  //
  // IEEE 754 standard float zero is compatible with setting four characters of
  // zero. memset should only be used to set bytes (characters) but since zero
  // is an exceptional case, a contiguous block is made zero when decltype(T) is
  // float or double.
  memset(c, 0, sizeof(T) * m * n);

#ifdef NO_NEON
  for (index_t ib = 0; ib < m; ib += i_block_size) {
    /* i block overflow condition */
    index_t ib_lim = std::min(ib + i_block_size, m);
    for (index_t jb = 0; jb < n; jb += j_block_size) {
      /* j block overflow condition */
      index_t jb_lim = std::min(jb + j_block_size, n);
      for (index_t kb = 0; kb < k; kb += k_block_size) {
        /* k block overflow condition */
        index_t kb_lim = std::min(kb + k_block_size, k);
        /* compute block matrix result */
        for (index_t ii = ib; ii < ib_lim; ++ii) {
          for (index_t jj = jb; jj < jb_lim; ++jj) {
            for (index_t kk = kb; kk < kb_lim; ++kk) {
              c[ii * n + jj] +=
                  a[(kk * m + ii) * a_t + (ii * k + kk) * (1 - a_t)] *
                  b[(jj * k + kk) * b_t + (kk * n + jj) * (1 - b_t)];
            }
          }
        }
      }
    }
  }
#else

#endif
  return 1;
}

#endif