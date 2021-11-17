#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <x86intrin.h>
#include "mkl.h"
using namespace std;

float G[6][3] = {
    {1./4., 0.0, 0.0},
    {-1./6., -1./6., -1./6.},
    {-1./6., 1./6., -1./6.},
    {1./24., 1./12., 1./6.},
    {1./24., -1./12., 1./6.},
    {0., 0., 1.}
  };
float G_T[3][6] = {
    {1./4., -1./6., -1./6., 1./24., 1./24., 0.0},
    {0.0, -1./6., 1./6., 1./12., -1./12., 0.0},
    {0.0, -1./6., -1./6., 1./6., 1./6., 1.0}
  };
float B[6][6] = {
    {4, 0, 0, 0, 0, 0},
    {0, -4, 4, -2, 2, 4},
    {-5, -4, -4, -1, -1, 0},
    {0, 1, -1, 2, -2, -5},
    {1, 1, 1, 1, 1, 0},
    {0, 0, 0, 0, 0, 1}
  };
float B_T[6][6] = {
    {4, 0, -5, 0, 1, 0},
    {0, -4, -4, 1, 1, 0},
    {0, 4, -4, -1, 1, 0},
    {0, -2, -1, 2, 1, 0},
    {0, 2, -1, -2, 1, 0},
    {0, 4, 0, -5, 0, 1}
  };
float A[6][4] = {
  {1, 0, 0, 0},
  {1, 1, 1, 1},
  {1, -1, 1, -1},
  {1, 2, 4, 8},
  {1, -2, 4, -8},
  {0, 0, 0, 1}
};
float A_T[4][6] = {
  {1, 1, 1, 1, 1, 0},
  {0, 1, -1, 2, -2, 0},
  {0, 1, 1, 4, 4, 0},
  {0, 1, -1, 8, -8, 1}
};

// Matrix Multiplication: Out = A x B (A:M*K, B:K*N, out: M*N)
// All arrays should have their memory prepared correctly outside this function
// For rookies: this sgemm is the worst sgemm I've ever written throughout my
// career.
//      If you don't know where to start, optimize this function as a good
//      starting point.
/*
void sgemm(const float *A, const float *B, float *out, const int M, const int K, const int N) {
  for (int i = 0; i < M * N; ++i) {
    out[i] = 0.0f;
  }
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < M; ++i)
          out[i * N + j]  += A[i * K + k] * B[k * N + j];
}
*/

void winconv_4x3(float *__restrict__ image, const int inHeight,
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
  const int trueHeight = outHeight % 4 == 0 ? outHeight / 4 : outHeight / 4 + 1;
  const int trueWidth  = outWidth % 4 == 0 ? outWidth / 4 : outWidth / 4 + 1;
  const int P = trueHeight * trueWidth * N;

  void * jitter_s_466, * jitter_s_464, * jitter_s_666, * jitter_s_633, * jitter_s_636;
  mkl_jit_status_t status;
  status = mkl_jit_create_sgemm(&jitter_s_466, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, 4, 6, 6, 1.0, 6, 6, 0.0, 6);
  if (MKL_JIT_ERROR == status){ printf("Creation jitter466 failed.\n"); }
  status = mkl_jit_create_sgemm(&jitter_s_464, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, 4, 4, 6, 1.0, 6, 4, 0.0, 4);
  if (MKL_JIT_ERROR == status){ printf("Creation jitter464 failed.\n"); }
  status = mkl_jit_create_sgemm(&jitter_s_666, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, 6, 6, 6, 1.0, 6, 6, 0.0, 6);
  if (MKL_JIT_ERROR == status){ printf("Creation jitter666 failed.\n"); }
  status = mkl_jit_create_sgemm(&jitter_s_633, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, 6, 3, 3, 1.0, 3, 3, 0.0, 3);
  if (MKL_JIT_ERROR == status){ printf("Creation jitter633 failed.\n"); }
  status = mkl_jit_create_sgemm(&jitter_s_636, MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, 6, 6, 3, 1.0, 3, 6, 0.0, 6);
  if (MKL_JIT_ERROR == status){ printf("Creation jitter636 failed.\n"); }

  sgemm_jit_kernel_t sgemm466 = mkl_jit_get_sgemm_ptr(jitter_s_466);
  sgemm_jit_kernel_t sgemm464 = mkl_jit_get_sgemm_ptr(jitter_s_464);
  sgemm_jit_kernel_t sgemm666 = mkl_jit_get_sgemm_ptr(jitter_s_666);
  sgemm_jit_kernel_t sgemm633 = mkl_jit_get_sgemm_ptr(jitter_s_633);
  sgemm_jit_kernel_t sgemm636 = mkl_jit_get_sgemm_ptr(jitter_s_636);

  float tmp_u[18];
  float u[36];
  #pragma omp parallel for private(tmp_u, u) collapse(2)
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      float *filters_ptr = filter + (k * C + c) * sizeF;
      sgemm633(jitter_s_633, &G[0][0], filters_ptr, tmp_u);
      sgemm636(jitter_s_636, tmp_u, &G_T[0][0], u);
      for (int xi = 0; xi < 6; ++xi)
        for (int nu = 0; nu < 6; ++nu)
          U[((xi * 6 + nu) * K + k) * C + c] = u[xi * 6 + nu];
    }
  }

  float tmp_v[36];
  float d[36];
  float v[36];

  #pragma omp parallel for private(tmp_v, d, v) collapse(4)
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < trueHeight; ++y) {
        for (int x = 0; x < trueWidth; ++x) {
          for (int iy = 0; iy < 6; ++iy)
            for (int ix = 0; ix < 6; ++ix){
              d[iy * 6 + ix] = (y * 4 + iy) >= inHeight || (x * 4 + ix) >= inWidth ? 0. :
                image[(n * C + c) * sizeI + (y * 4 + iy) * inWidth + (x * 4 + ix)];
            }

          sgemm666(jitter_s_666, &B_T[0][0], d, tmp_v);
          sgemm666(jitter_s_666, tmp_v, &B[0][0], v);

          int b = ((n * trueHeight) + y) * trueWidth + x;
          for (int xi = 0; xi < 6; ++xi)
            for (int nu = 0; nu < 6; ++nu)
              V[((xi * 6 + nu) * C + c) * P + b] = v[xi * 6 + nu];
        }
      }
    }

  #pragma omp parallel for collapse(2)
  for (int xi = 0; xi < 6; ++xi) {
    for (int nu = 0; nu < 6; ++nu) {
      float *M_ptr = M + (xi * 6 + nu) * K * P;
      float *U_ptr = U + (xi * 6 + nu) * K * C;
      float *V_ptr = V + (xi * 6 + nu) * C * P;
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, P, C, 1.0, U_ptr, C, V_ptr, P, 0.0, M_ptr, P);
    }
  }

  float mm[36];
  float temp_out[16];
  float tmp_m[24];
  #pragma omp parallel for private(mm, temp_out, tmp_m) collapse(4)
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      for (int y = 0; y < trueHeight; ++y) {
        for (int x = 0; x < trueWidth; ++x) {
          int b = (n * trueHeight + y) * trueWidth + x;
          for (int xi = 0; xi < 6; ++xi) {
            for (int nu = 0; nu < 6; ++nu) {
              mm[xi * 6 + nu] = M[((xi * 6 + nu) * K + k) * P + b];
            }
          }
          sgemm466(jitter_s_466, &A_T[0][0], mm, tmp_m);
          sgemm464(jitter_s_464, tmp_m, &A[0][0], temp_out);

          for (int i = 0; i < 4; ++i){
            if( (y * 4 + i) >= outHeight ) break;
            for (int j = 0; j < 4; ++j){
              if( (x * 4 + j) >= outWidth ) break;
              out[((n * K + k) * outHeight + y * 4 + i) * outWidth + x * 4 + j] = temp_out[i * 4 + j];
            }
          }
        }
      }
    }

    mkl_jit_destroy(jitter_s_466);
    mkl_jit_destroy(jitter_s_464);
    mkl_jit_destroy(jitter_s_666);
    mkl_jit_destroy(jitter_s_633);
    mkl_jit_destroy(jitter_s_636);
}
