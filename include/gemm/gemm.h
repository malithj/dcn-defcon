#ifndef __GEMM_H_
#define __GEMM_H_

#include "core/types/types.h"
#include "sgemm.h"

template <typename T>
class GEMM {
 public:
  GEMM(){};
  ~GEMM() = default;
  void sgemm(char transa, char transb, index_t m, index_t n, index_t k, T alpha,
             const T* a, index_t lda, const T* b, index_t ldb, T beta, T* c,
             index_t ldc);

 private:
  SGEMM<T> _gemm;
};

template <typename T>
void GEMM<T>::sgemm(char transa, char transb, index_t m, index_t n, index_t k,
                    T alpha, const T* a, index_t lda, const T* b, index_t ldb,
                    T beta, T* c, index_t ldc) {
  SGEMM<T> _gemm;
  _gemm.gemm(transa, transb, m, n, k, 1, a, k, b, n, 0, c, n);
}

#endif