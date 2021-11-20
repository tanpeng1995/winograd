#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <x86intrin.h>
#include "mkl.h"
using namespace std;

float G[8][3] = {
    {1.,      0.,       0.},
    {-2./9.,  -2./9.,   -2./9.},
    {-2./9.,  2./9.,    -2./9.},
    {1./90.,  1./45.,   2./45.},
    {1./90.,  -1./45.,  2./45.},
    {32./45., 16./45.,  8./45.},
    {32./45., -16./45., 8./45.},
    {0.,      0.,       1.}
  };
float G_T[3][8] = {
    {1., -2./9., -2./9., 1./90., 1./90., 32./45., 32./45., 0.},
    {0., -2./9., 2./9., 1./45., -1./45., 16./45., -16./45., 0.},
    {0., -2./9., -2./9., 2./45., 2./45., 8./45., 8./45., 1.}
  };
float B[8][8] = {
    {1., 0., 0., 0., 0., 0., 0., 0.},
    {0., 1., -1., 1./2., -1./2., 2., -2., -1.},
    {-21./4., 1., 1., 1./4., 1./4., 4., 4., 0.},
    {0., -17./4., 17./4., -5./2., 5./2., -5./2., 5./2., 21./4.},
    {21./4., -17./4., -17./4., -5./4., -5./4., -5., -5., 0.},
    {0., 1., -1., 2., -2., 1./2., -1./2., -21./4.},
    {-1., 1., 1., 1., 1., 1., 1., 0.},
    {0., 0., 0., 0., 0., 0., 0., 1.}
  };
float B_T[8][8] = {
    {1.,   0.,     -21./4.,   0,        21./4.,    0.,     -1., 0.},
    {0.,   1.,     1.,        -17./4.,  -17./4.,   1.,      1., 0.},
    {0.,   -1.,    1.,        17./4.,   -17./4.,   -1.,     1., 0.},
    {0.,   1./2.,  1./4.,     -5./2.,   -5./4.,    2.,      1., 0.},
    {0.,   -1./2., 1./4.,     5./2.,    -5./4.,    -2.,     1., 0.},
    {0.,   2.,     4.,        -5./2.,   -5.,       1./2.,   1., 0.},
    {0.,   -2.,    4.,        5./2.,    -5.,       -1./2.,  1., 0.},
    {0.,   -1.,    0.,        21./4.,   0.,        -21./4., 0., 1.}
  };
float A[8][6] = {
  {1., 0.,      0.,     0.,     0.,     0.},
  {1., 1.,      1.,     1.,     1.,     1.},
  {1., -1.,     1.,     -1.,    1.,     -1.},
  {1., 2.,      4.,     8.,     16.,    32.},
  {1., -2.,     4.,     -8.,    16.,    -32.},
  {1., 1./2.,   1./4.,  1./8.,  1./16., 1./32.},
  {1., -1./2.,  1./4.,  -1./8., 1./16., -1./32.},
  {0., 0.,      0.,     0.,     0.,     1.}
};
float A_T[6][8] = {
  {1., 1., 1.,  1.,  1.,   1.,     1.,     0.},
  {0., 1., -1., 2.,  -2.,  1./2.,  -1./2., 0.},
  {0., 1., 1.,  4.,  4.,   1./4.,  1./4.,  0.},
  {0., 1., -1., 8.,  -8.,  1./8.,  -1./8., 0.},
  {0., 1., 1.,  16., 16.,  1./16., 1./16., 0.},
  {0., 1., -1., 32., -32., 1./32., -1./32., 1.}
};

void winconv_6x3(float *__restrict__ image, const int inHeight,
                 const int inWidth, const int C, float *__restrict__ filter,
                 const int K, const int N, float *__restrict__ out,
                 float *__restrict__ U, float *__restrict__ V,
                 float *__restrict__ M) {
  // m = 4; r = 3; alpha = 6
  const int outHeight = inHeight - 2;
  const int outWidth = inWidth - 2;
  const int sizeI = inHeight * inWidth;
  const int sizeF = 3 * 3;
  const int sizeO = outHeight * outWidth;
  const int trueHeight = outHeight % 6 == 0 ? outHeight / 6 : outHeight / 6 + 1;
  const int trueWidth  = outWidth % 6 == 0 ? outWidth / 6 : outWidth / 6 + 1;
  const int P = trueHeight * trueWidth * N;

  void * jitter_833, * jitter_838, * jitter_888, * jitter_688, * jitter_686;
  mkl_jit_status_t status;
  status = mkl_jit_create_sgemm(&jitter_833, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, 8, 3, 3, 1.0, 3, 3, 0.0, 3);
  if (MKL_JIT_ERROR == status){ printf("Creation jitter_833 failed.\n"); }
  status = mkl_jit_create_sgemm(&jitter_838, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, 8, 8, 3, 1.0, 3, 8, 0.0, 8);
  if (MKL_JIT_ERROR == status){ printf("Creation jitter_838 failed.\n"); }
  status = mkl_jit_create_sgemm(&jitter_888, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, 8, 8, 8, 1.0, 8, 8, 0.0, 8);
  if (MKL_JIT_ERROR == status){ printf("Creation jitter_888 failed.\n"); }
  status = mkl_jit_create_sgemm(&jitter_688, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, 6, 8, 8, 1.0, 8, 8, 0.0, 8);
  if (MKL_JIT_ERROR == status){ printf("Creation jitter_688 failed.\n"); }
  status = mkl_jit_create_sgemm(&jitter_686, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, 6, 6, 8, 1.0, 8, 6, 0.0, 6);
  if (MKL_JIT_ERROR == status){ printf("Creation jitter_686 failed.\n"); }

  sgemm_jit_kernel_t sgemm833 = mkl_jit_get_sgemm_ptr(jitter_833);
  sgemm_jit_kernel_t sgemm838 = mkl_jit_get_sgemm_ptr(jitter_838);
  sgemm_jit_kernel_t sgemm888 = mkl_jit_get_sgemm_ptr(jitter_888);
  sgemm_jit_kernel_t sgemm688 = mkl_jit_get_sgemm_ptr(jitter_688);
  sgemm_jit_kernel_t sgemm686 = mkl_jit_get_sgemm_ptr(jitter_686);

  float tmp_u[24];
  float u[64];
  #pragma omp parallel for private(tmp_u, u) collapse(2)
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      float *g = filter + (k * C + c) * sizeF;
      sgemm833(jitter_833, &G[0][0], g, tmp_u);
      sgemm838(jitter_838, tmp_u, &G_T[0][0], u);
      for (int xi = 0; xi < 8; ++xi)
        for (int nu = 0; nu < 8; ++nu)
          U[((xi * 8 + nu) * K + k) * C + c] = u[xi * 8 + nu];
    }
  }

  float tmp_v[64];
  float d[64];
  float v[64];
  #pragma omp parallel for private(tmp_v, d, v) collapse(4)
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < trueHeight; ++y) {
        for (int x = 0; x < trueWidth; ++x) {
          for (int iy = 0; iy < 8; ++iy)
            for (int ix = 0; ix < 8; ++ix){
              d[iy * 8 + ix] = (y * 6 + iy) >= inHeight || (x * 6 + ix) >= inWidth ? 0. :
                image[(n * C + c) * sizeI + (y * 6 + iy) * inWidth + (x * 6 + ix)];
            }

          sgemm888(jitter_888, &B_T[0][0], d, tmp_v);
          sgemm888(jitter_888, tmp_v, &B[0][0], v);

          int b = ((n * trueHeight) + y) * trueWidth + x;
          for (int xi = 0; xi < 8; ++xi)
            for (int nu = 0; nu < 8; ++nu)
              V[((xi * 8 + nu) * C + c) * P + b] = v[xi * 8 + nu];
        }
      }
    }

/*
  #pragma omp parallel for collapse(2)
  for (int xi = 0; xi < 8; ++xi) {
    for (int nu = 0; nu < 8; ++nu) {
      float *M_ptr = M + (xi * 8 + nu) * K * P;
      float *U_ptr = U + (xi * 8 + nu) * K * C;
      float *V_ptr = V + (xi * 8 + nu) * C * P;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, P, C, 1.0, U_ptr, C, V_ptr, P, 0.0, M_ptr, P);
    }
  }
  */
  MKL_INT GRP_COUNT = 1;
  const MKL_INT m[GRP_COUNT] = {K};
  const MKL_INT n[GRP_COUNT] = {P};
  const MKL_INT k[GRP_COUNT] = {C};
  const MKL_INT lda[GRP_COUNT] = {C};
  const MKL_INT ldb[GRP_COUNT] = {P};
  const MKL_INT ldc[GRP_COUNT] = {P};
  const CBLAS_TRANSPOSE transA[GRP_COUNT] = {CblasNoTrans};
  const CBLAS_TRANSPOSE transB[GRP_COUNT] = {CblasNoTrans};
  const float alpha[GRP_COUNT] = {1.0};
  const float beta[GRP_COUNT] = {0.0};
  const MKL_INT size_per_grp[GRP_COUNT] = {64};
  const float *a_array[64], *b_array[64];
  float *c_array[64];
  #pragma omp parallel for collapse(2)
  for (int xi = 0; xi < 8; ++xi) {
    for (int nu = 0; nu < 8; ++nu) {
      c_array[xi * 8 + nu] = M + (xi * 8 + nu) * K * P;
      a_array[xi * 8 + nu] = U + (xi * 8 + nu) * K * C;
      b_array[xi * 8 + nu] = V + (xi * 8 + nu) * C * P;
    }
  }
  cblas_sgemm_batch(CblasRowMajor, transA, transB,
    m, n, k, alpha,
    a_array, lda,
    b_array, ldb, beta,
    c_array, ldc,
    GRP_COUNT, size_per_grp);


  float mm[64];
  float temp_out[36];
  float tmp_m[48];
  #pragma omp parallel for private(mm, temp_out, tmp_m) collapse(4)
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      for (int y = 0; y < trueHeight; ++y) {
        for (int x = 0; x < trueWidth; ++x) {
          int b = (n * trueHeight + y) * trueWidth + x;
          for (int xi = 0; xi < 8; ++xi) {
            for (int nu = 0; nu < 8; ++nu) {
              mm[xi * 8 + nu] = M[((xi * 8 + nu) * K + k) * P + b];
            }
          }
          sgemm688(jitter_688, &A_T[0][0], mm, tmp_m);
          sgemm686(jitter_686, tmp_m, &A[0][0], temp_out);

          for (int i = 0; i < 6; ++i){
            if( (y * 6 + i) >= outHeight ) break;
            for (int j = 0; j < 6; ++j){
              if( (x * 6 + j) >= outWidth ) break;
              out[((n * K + k) * outHeight + y * 6 + i) * outWidth + x * 6 + j] = temp_out[i * 6 + j];
            }
          }
        }
      }
    }

    mkl_jit_destroy(jitter_833);
    mkl_jit_destroy(jitter_838);
    mkl_jit_destroy(jitter_888);
    mkl_jit_destroy(jitter_688);
    mkl_jit_destroy(jitter_686);
}
