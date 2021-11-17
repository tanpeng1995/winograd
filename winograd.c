#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <mkl.h>

const float G[4][3] = {
    {1.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0.0, 0.0, 1.0}};
const float G_T[3][4] = {
    {1, 0.5, 0.5, 0.0}, {0.0, 0.5, -0.5, 0.0}, {0.0, 0.5, 0.5, 1.0}};
const float B[4][4] = {
    {1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};
const float B_T[4][4] = {
    {1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
const float A[4][2] = {{1, 0}, {1, 1}, {1, -1}, {0, -1}};
const float A_T[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};

#define L1_DIST_A 24
#define L1_DIST_B 24

// Matrix Multiplication: Out = A x B (A:M*K, B:K*N, out: M*N)
// All arrays should have their memory prepared correctly outside this function
// For rookies: this sgemm is the worst sgemm I've ever written throughout my
// career.
//      If you don't know where to start, optimize this function as a good
//      starting point.
void sgemm(const float *A, const float *B, float *out, const int M, const int K, const int N) {
  for (int i = 0; i < M * N; ++i) {
    out[i] = 0.0f;
  }
  for (int k = 0; k < K; ++k)
    for (int j = 0; j < N; ++j)
      for (int i = 0; i < M; ++i)
          out[i * N + j]  += A[i * K + k] * B[k * N + j];
}

void sgemm_with_kernel(const float *A, const float *B, float *out, const int M, const int K, const int N) {
  #pragma omp parallel for
  for (int i = 0; i < M / 8 * 8; i += 8){
    for (int j = 0; j < N / 16 * 16; j += 16){
      register __m512 _C0 = _mm512_setzero_ps();
      register __m512 _C1 = _mm512_setzero_ps();
      register __m512 _C2 = _mm512_setzero_ps();
      register __m512 _C3 = _mm512_setzero_ps();
      register __m512 _C4 = _mm512_setzero_ps();
      register __m512 _C5 = _mm512_setzero_ps();
      register __m512 _C6 = _mm512_setzero_ps();
      register __m512 _C7 = _mm512_setzero_ps();
      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m512 _B = _mm512_loadu_ps(B+N*k+j);
        _C0 = _mm512_fmadd_ps(_mm512_set1_ps(A[i*K+k]),     _B, _C0);
        _C1 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i+1)*K+k]), _B, _C1);
        _C2 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i+2)*K+k]), _B, _C2);
        _C3 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i+3)*K+k]), _B, _C3);
        _C4 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i+4)*K+k]), _B, _C4);
        _C5 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i+5)*K+k]), _B, _C5);
        _C6 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i+6)*K+k]), _B, _C6);
        _C7 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i+7)*K+k]), _B, _C7);

      }
      _mm512_storeu_ps(out+(i)*N+j,     _C0);
      _mm512_storeu_ps(out+(i+1)*N+j,   _C1);
      _mm512_storeu_ps(out+(i+2)*N+j,   _C2);
      _mm512_storeu_ps(out+(i+3)*N+j,   _C3);
      _mm512_storeu_ps(out+(i+4)*N+j,   _C4);
      _mm512_storeu_ps(out+(i+5)*N+j,   _C5);
      _mm512_storeu_ps(out+(i+6)*N+j,   _C6);
      _mm512_storeu_ps(out+(i+7)*N+j,   _C7);
    }
    for (int j = N / 16 * 16; j < N / 8 * 8; j += 8){
      register __m256 _C0 = _mm256_setzero_ps();
      register __m256 _C1 = _mm256_setzero_ps();
      register __m256 _C2 = _mm256_setzero_ps();
      register __m256 _C3 = _mm256_setzero_ps();
      register __m256 _C4 = _mm256_setzero_ps();
      register __m256 _C5 = _mm256_setzero_ps();
      register __m256 _C6 = _mm256_setzero_ps();
      register __m256 _C7 = _mm256_setzero_ps();

      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m256 _B = _mm256_loadu_ps(B+N*k+j);
        _C0 = _mm256_fmadd_ps(_mm256_set1_ps(A[i*K+k]),     _B, _C0);
        _C1 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i+1)*K+k]), _B, _C1);
        _C2 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i+2)*K+k]), _B, _C2);
        _C3 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i+3)*K+k]), _B, _C3);
        _C4 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i+4)*K+k]), _B, _C4);
        _C5 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i+5)*K+k]), _B, _C5);
        _C6 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i+6)*K+k]), _B, _C6);
        _C7 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i+7)*K+k]), _B, _C7);
      }
      _mm256_storeu_ps(out+(i)*N+j,     _C0);
      _mm256_storeu_ps(out+(i+1)*N+j,   _C1);
      _mm256_storeu_ps(out+(i+2)*N+j,   _C2);
      _mm256_storeu_ps(out+(i+3)*N+j,   _C3);
      _mm256_storeu_ps(out+(i+4)*N+j,   _C4);
      _mm256_storeu_ps(out+(i+5)*N+j,   _C5);
      _mm256_storeu_ps(out+(i+6)*N+j,   _C6);
      _mm256_storeu_ps(out+(i+7)*N+j,   _C7);
    }
    for (int j = N / 8 * 8; j < N / 4 * 4; j += 4){
      register __m128 _C0 = _mm_setzero_ps();
      register __m128 _C1 = _mm_setzero_ps();
      register __m128 _C2 = _mm_setzero_ps();
      register __m128 _C3 = _mm_setzero_ps();
      register __m128 _C4 = _mm_setzero_ps();
      register __m128 _C5 = _mm_setzero_ps();
      register __m128 _C6 = _mm_setzero_ps();
      register __m128 _C7 = _mm_setzero_ps();

      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m128 _B = _mm_loadu_ps(B+N*k+j);
        _C0 = _mm_fmadd_ps(_mm_set1_ps(A[(i)*K+k]), _B, _C0);
        _C1 = _mm_fmadd_ps(_mm_set1_ps(A[(i+1)*K+k]), _B, _C1);
        _C2 = _mm_fmadd_ps(_mm_set1_ps(A[(i+2)*K+k]), _B, _C2);
        _C3 = _mm_fmadd_ps(_mm_set1_ps(A[(i+3)*K+k]), _B, _C3);
        _C4 = _mm_fmadd_ps(_mm_set1_ps(A[(i+4)*K+k]), _B, _C4);
        _C5 = _mm_fmadd_ps(_mm_set1_ps(A[(i+5)*K+k]), _B, _C5);
        _C6 = _mm_fmadd_ps(_mm_set1_ps(A[(i+6)*K+k]), _B, _C6);
        _C7 = _mm_fmadd_ps(_mm_set1_ps(A[(i+7)*K+k]), _B, _C7);
      }
      _mm_storeu_ps(out+(i)*N+j,   _C0);
      _mm_storeu_ps(out+(i+1)*N+j, _C1);
      _mm_storeu_ps(out+(i+2)*N+j, _C2);
      _mm_storeu_ps(out+(i+3)*N+j, _C3);
      _mm_storeu_ps(out+(i+4)*N+j, _C4);
      _mm_storeu_ps(out+(i+5)*N+j, _C5);
      _mm_storeu_ps(out+(i+6)*N+j, _C6);
      _mm_storeu_ps(out+(i+7)*N+j, _C7);
    }

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < K; k += 1){
      for (int j = N / 4 * 4; j < N; j += 1){
        out[i*N+j]     += A[i*K+k]     * B[k*N+j];
        out[(i+1)*N+j] += A[(i+1)*K+k] * B[k*N+j];
        out[(i+2)*N+j] += A[(i+2)*K+k] * B[k*N+j];
        out[(i+3)*N+j] += A[(i+3)*K+k] * B[k*N+j];
        out[(i+4)*N+j] += A[(i+4)*K+k] * B[k*N+j];
        out[(i+5)*N+j] += A[(i+5)*K+k] * B[k*N+j];
        out[(i+6)*N+j] += A[(i+6)*K+k] * B[k*N+j];
        out[(i+7)*N+j] += A[(i+7)*K+k] * B[k*N+j];
      }
    }
  }
  #pragma omp parallel for
  for (int i = M / 8 * 8; i < M / 4 * 4; i += 4){
    for (int j = 0; j < N / 16 * 16; j += 16){
      register __m512 _C0 = _mm512_setzero_ps();
      register __m512 _C1 = _mm512_setzero_ps();
      register __m512 _C2 = _mm512_setzero_ps();
      register __m512 _C3 = _mm512_setzero_ps();

      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m512 _B = _mm512_loadu_ps(B+N*k+j);
        _C0 = _mm512_fmadd_ps(_mm512_set1_ps(A[i*K+k]),     _B, _C0);
        _C1 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i+1)*K+k]), _B, _C1);
        _C2 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i+2)*K+k]), _B, _C2);
        _C3 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i+3)*K+k]), _B, _C3);
      }
      _mm512_storeu_ps(out+(i)*N+j,     _C0);
      _mm512_storeu_ps(out+(i+1)*N+j,   _C1);
      _mm512_storeu_ps(out+(i+2)*N+j,   _C2);
      _mm512_storeu_ps(out+(i+3)*N+j,   _C3);
    }

    for (int j = N / 16 * 16; j < N / 8 * 8; j += 8){
      register __m256 _C0 = _mm256_setzero_ps();
      register __m256 _C1 = _mm256_setzero_ps();
      register __m256 _C2 = _mm256_setzero_ps();
      register __m256 _C3 = _mm256_setzero_ps();

      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m256 _B = _mm256_loadu_ps(B+N*k+j);
        _C0 = _mm256_fmadd_ps(_mm256_set1_ps(A[i*K+k]),     _B, _C0);
        _C1 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i+1)*K+k]), _B, _C1);
        _C2 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i+2)*K+k]), _B, _C2);
        _C3 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i+3)*K+k]), _B, _C3);
      }
      _mm256_storeu_ps(out+(i)*N+j,     _C0);
      _mm256_storeu_ps(out+(i+1)*N+j,   _C1);
      _mm256_storeu_ps(out+(i+2)*N+j,   _C2);
      _mm256_storeu_ps(out+(i+3)*N+j,   _C3);
    }

    for (int j = N / 8 * 8; j < N / 4 * 4; j += 4){
      register __m128 _C0 = _mm_setzero_ps();
      register __m128 _C1 = _mm_setzero_ps();
      register __m128 _C2 = _mm_setzero_ps();
      register __m128 _C3 = _mm_setzero_ps();

      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m128 _B = _mm_loadu_ps(B+N*k+j);
        _C0 = _mm_fmadd_ps(_mm_set1_ps(A[(i)*K+k]), _B, _C0);
        _C1 = _mm_fmadd_ps(_mm_set1_ps(A[(i+1)*K+k]), _B, _C1);
        _C2 = _mm_fmadd_ps(_mm_set1_ps(A[(i+2)*K+k]), _B, _C2);
        _C3 = _mm_fmadd_ps(_mm_set1_ps(A[(i+3)*K+k]), _B, _C3);
      }
      _mm_storeu_ps(out+(i)*N+j,   _C0);
      _mm_storeu_ps(out+(i+1)*N+j, _C1);
      _mm_storeu_ps(out+(i+2)*N+j, _C2);
      _mm_storeu_ps(out+(i+3)*N+j, _C3);
    }

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < K; k += 1){
      for (int j = N / 4 * 4; j < N; j += 1){
        out[i*N+j]     += A[i*K+k]     * B[k*N+j];
        out[(i+1)*N+j] += A[(i+1)*K+k] * B[k*N+j];
        out[(i+2)*N+j] += A[(i+2)*K+k] * B[k*N+j];
        out[(i+3)*N+j] += A[(i+3)*K+k] * B[k*N+j];
      }
    }
  }
  #pragma omp parallel for
  for (int i = M / 4 * 4; i < M / 2 * 2; i += 2){
    for (int j = 0; j < N / 16 * 16; j += 16){
      register __m512 _C0 = _mm512_setzero_ps();
      register __m512 _C1 = _mm512_setzero_ps();
      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m512 _B = _mm512_loadu_ps(B+N*k+j);
        _C0 = _mm512_fmadd_ps(_mm512_set1_ps(A[i*K+k]),     _B, _C0);
        _C1 = _mm512_fmadd_ps(_mm512_set1_ps(A[(i+1)*K+k]), _B, _C1);
      }
      _mm512_storeu_ps(out+(i)*N+j,     _C0);
      _mm512_storeu_ps(out+(i+1)*N+j,   _C1);
    }

    for (int j = N / 16 * 16; j < N / 8 * 8; j += 8){
      register __m256 _C0 = _mm256_setzero_ps();
      register __m256 _C1 = _mm256_setzero_ps();
      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m256 _B = _mm256_loadu_ps(B+N*k+j);
        _C0 = _mm256_fmadd_ps(_mm256_set1_ps(A[i*K+k]),     _B, _C0);
        _C1 = _mm256_fmadd_ps(_mm256_set1_ps(A[(i+1)*K+k]), _B, _C1);
      }
      _mm256_storeu_ps(out+(i)*N+j,     _C0);
      _mm256_storeu_ps(out+(i+1)*N+j,   _C1);
    }

    for (int j = N / 8 * 8; j < N / 4 * 4; j += 4){
      register __m128 _C0 = _mm_setzero_ps();
      register __m128 _C1 = _mm_setzero_ps();
      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m128 _B = _mm_loadu_ps(B+N*k+j);
        _C0 = _mm_fmadd_ps(_mm_set1_ps(A[(i)*K+k]), _B, _C0);
        _C1 = _mm_fmadd_ps(_mm_set1_ps(A[(i+1)*K+k]), _B, _C1);
      }
      _mm_storeu_ps(out+(i)*N+j,   _C0);
      _mm_storeu_ps(out+(i+1)*N+j, _C1);
    }

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < K; k += 1){
      for (int j = N / 4 * 4; j < N; j += 1){
        out[i*N+j]     += A[i*K+k]     * B[k*N+j];
        out[(i+1)*N+j] += A[(i+1)*K+k] * B[k*N+j];
      }
    }
  }
  #pragma omp parallel for
  for (int i = M / 2 * 2; i < M; i += 1){
    for (int j = 0; j < N / 16 * 16; j += 16){
      register __m512 _C0 = _mm512_setzero_ps();
      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m512 _B = _mm512_loadu_ps(B+N*k+j);
        _C0 = _mm512_fmadd_ps(_mm512_set1_ps(A[i*K+k]),     _B, _C0);
      }
      _mm512_storeu_ps(out+(i)*N+j,     _C0);
    }

    for (int j = N / 16 * 16; j < N / 8 * 8; j += 8){
      register __m256 _C0 = _mm256_setzero_ps();
      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m256 _B = _mm256_loadu_ps(B+N*k+j);
        _C0 = _mm256_fmadd_ps(_mm256_set1_ps(A[i*K+k]),     _B, _C0);
      }
      _mm256_storeu_ps(out+(i)*N+j,     _C0);
    }

    for (int j = N / 8 * 8; j < N / 4 * 4; j += 4){
      register __m128 _C0 = _mm_setzero_ps();
      for (int k = 0; k < K; k += 1){
        _mm_prefetch((char*) &B[N*k+j+L1_DIST_B],_MM_HINT_T0);
        register __m128 _B = _mm_loadu_ps(B+N*k+j);
        _C0 = _mm_fmadd_ps(_mm_set1_ps(A[(i)*K+k]), _B, _C0);
      }
      _mm_storeu_ps(out+(i)*N+j,   _C0);
    }

    #pragma omp parallel for collapse(2)
    for (int k = 0; k < K; k += 1){
      for (int j = N / 4 * 4; j < N; j += 1){
        out[i*N+j]     += A[i*K+k] * B[k*N+j];
      }
    }
  }
}

void sgemm434(const float *A, const float *B, float *out){
  //M = 4; K = 3; N = 4; out = 4x4
  register __m128 _C0, _C1, _C2, _C3;
  _C0 = _mm_setzero_ps();
  _C1 = _mm_setzero_ps();
  _C2 = _mm_setzero_ps();
  _C3 = _mm_setzero_ps();

  register __m128 _B0, _B1, _B2;
  _B0 = _mm_loadu_ps(&B[0]);
  _B1 = _mm_loadu_ps(&B[4]);
  _B2 = _mm_loadu_ps(&B[8]);

  _C0 = _mm_fmadd_ps(_mm_set1_ps(A[0]), _B0, _C0);
  _C0 = _mm_fmadd_ps(_mm_set1_ps(A[1]), _B1, _C0);
  _C0 = _mm_fmadd_ps(_mm_set1_ps(A[2]), _B2, _C0);

  _C1 = _mm_fmadd_ps(_mm_set1_ps(A[3]), _B0, _C1);
  _C1 = _mm_fmadd_ps(_mm_set1_ps(A[4]), _B1, _C1);
  _C1 = _mm_fmadd_ps(_mm_set1_ps(A[5]), _B2, _C1);

  _C2 = _mm_fmadd_ps(_mm_set1_ps(A[6]), _B0, _C2);
  _C2 = _mm_fmadd_ps(_mm_set1_ps(A[7]), _B1, _C2);
  _C2 = _mm_fmadd_ps(_mm_set1_ps(A[8]), _B2, _C2);

  _C3 = _mm_fmadd_ps(_mm_set1_ps(A[9]), _B0, _C3);
  _C3 = _mm_fmadd_ps(_mm_set1_ps(A[10]), _B1, _C3);
  _C3 = _mm_fmadd_ps(_mm_set1_ps(A[11]), _B2, _C3);

  _mm_storeu_ps(&out[0], _C0);
  _mm_storeu_ps(&out[4], _C1);
  _mm_storeu_ps(&out[8], _C2);
  _mm_storeu_ps(&out[12], _C3);
}

void sgemm444(const float *A, const float *B, float *out){
  //M = 4; K = 4; N = 4; out = 4x4
  register __m128 _C0, _C1, _C2, _C3;
  _C0 = _mm_setzero_ps();
  _C1 = _mm_setzero_ps();
  _C2 = _mm_setzero_ps();
  _C3 = _mm_setzero_ps();

  register __m128 _B0, _B1, _B2, _B3;
  _B0 = _mm_loadu_ps(&B[0]);
  _B1 = _mm_loadu_ps(&B[4]);
  _B2 = _mm_loadu_ps(&B[8]);
  _B3 = _mm_loadu_ps(&B[12]);

  _C0 = _mm_fmadd_ps(_mm_set1_ps(A[0]), _B0, _C0);
  _C0 = _mm_fmadd_ps(_mm_set1_ps(A[1]), _B1, _C0);
  _C0 = _mm_fmadd_ps(_mm_set1_ps(A[2]), _B2, _C0);
  _C0 = _mm_fmadd_ps(_mm_set1_ps(A[3]), _B3, _C0);

  _C1 = _mm_fmadd_ps(_mm_set1_ps(A[4]), _B0, _C1);
  _C1 = _mm_fmadd_ps(_mm_set1_ps(A[5]), _B1, _C1);
  _C1 = _mm_fmadd_ps(_mm_set1_ps(A[6]), _B2, _C1);
  _C1 = _mm_fmadd_ps(_mm_set1_ps(A[7]), _B3, _C1);

  _C2 = _mm_fmadd_ps(_mm_set1_ps(A[8]), _B0, _C2);
  _C2 = _mm_fmadd_ps(_mm_set1_ps(A[9]), _B1, _C2);
  _C2 = _mm_fmadd_ps(_mm_set1_ps(A[10]), _B2, _C2);
  _C2 = _mm_fmadd_ps(_mm_set1_ps(A[11]), _B3, _C2);

  _C3 = _mm_fmadd_ps(_mm_set1_ps(A[12]), _B0, _C3);
  _C3 = _mm_fmadd_ps(_mm_set1_ps(A[13]), _B1, _C3);
  _C3 = _mm_fmadd_ps(_mm_set1_ps(A[14]), _B2, _C3);
  _C3 = _mm_fmadd_ps(_mm_set1_ps(A[15]), _B3, _C3);

  _mm_storeu_ps(&out[0], _C0);
  _mm_storeu_ps(&out[4], _C1);
  _mm_storeu_ps(&out[8], _C2);
  _mm_storeu_ps(&out[12], _C3);
}

void sgemm244(const float *A, const float *B, float *out){
  //M = 2; K = 4; N = 4; out = 2x4
  register __m128 _C0, _C1;
  _C0 = _mm_setzero_ps();
  _C1 = _mm_setzero_ps();

  register __m128 _B0, _B1, _B2, _B3;
  _B0 = _mm_loadu_ps(&B[0]);
  _B1 = _mm_loadu_ps(&B[4]);
  _B2 = _mm_loadu_ps(&B[8]);
  _B3 = _mm_loadu_ps(&B[12]);

  _C0 = _mm_fmadd_ps(_mm_set1_ps(A[0]), _B0, _C0);
  _C0 = _mm_fmadd_ps(_mm_set1_ps(A[1]), _B1, _C0);
  _C0 = _mm_fmadd_ps(_mm_set1_ps(A[2]), _B2, _C0);
  _C0 = _mm_fmadd_ps(_mm_set1_ps(A[3]), _B3, _C0);

  _C1 = _mm_fmadd_ps(_mm_set1_ps(A[4]), _B0, _C1);
  _C1 = _mm_fmadd_ps(_mm_set1_ps(A[5]), _B1, _C1);
  _C1 = _mm_fmadd_ps(_mm_set1_ps(A[6]), _B2, _C1);
  _C1 = _mm_fmadd_ps(_mm_set1_ps(A[7]), _B3, _C1);

  _mm_storeu_ps(&out[0], _C0);
  _mm_storeu_ps(&out[4], _C1);
}

void sgemm242(const float *A, const float *B, float *out){
  //M = 2; K = 4; N = 2; out = 2x2
  out[0] = A[0]*B[0] + A[1]*B[2] + A[2]*B[4] + A[3]*B[6];
  out[1] = A[0]*B[1] + A[1]*B[3] + A[2]*B[5] + A[3]*B[7];
  out[2] = A[4]*B[0] + A[5]*B[2] + A[6]*B[4] + A[7]*B[6];
  out[3] = A[4]*B[1] + A[5]*B[3] + A[6]*B[5] + A[7]*B[7];
}

void sgemm433(const float *A, const float *B, float *out){
  //M = 4; K = 3; N = 3; out = 4x3
  out[0] = A[0]*B[0] + A[1]*B[3] + A[2]*B[6];
  out[1] = A[0]*B[1] + A[1]*B[4] + A[2]*B[7];
  out[2] = A[0]*B[2] + A[1]*B[5] + A[2]*B[8];

  out[3] = A[3]*B[0] + A[4]*B[3] + A[5]*B[6];
  out[4] = A[3]*B[1] + A[4]*B[4] + A[5]*B[7];
  out[5] = A[3]*B[2] + A[4]*B[5] + A[5]*B[8];

  out[6] = A[6]*B[0] + A[7]*B[3] + A[8]*B[6];
  out[7] = A[6]*B[1] + A[7]*B[4] + A[8]*B[7];
  out[8] = A[6]*B[2] + A[7]*B[5] + A[8]*B[8];

  out[9] = A[9]*B[0] + A[10]*B[3] + A[11]*B[6];
  out[10] = A[9]*B[1] + A[10]*B[4] + A[11]*B[7];
  out[11] = A[9]*B[2] + A[10]*B[5] + A[11]*B[8];
}

// User API for winograd F(2,3)
// image: [batch * C * inHeight * inWidth]
// filter: [K * C * 3 * 3]
// result: [batch * K * outHeight * outWidth]
void winconv_2x3(float *__restrict__ image, const int inHeight,
                 const int inWidth, const int C, float *__restrict__ filter,
                 const int K, const int N, float *__restrict__ out,
                 float *__restrict__ U, float *__restrict__ V,
                 float *__restrict__ M) {
  // m = 2; r = 3; alpha = 4
  const int outHeight = inHeight - 2;
  const int outWidth = inWidth - 2;
  const int sizeI = inHeight * inWidth;
  const int sizeF = 3 * 3;
  const int sizeO = outHeight * outWidth;
  const int P = outHeight / 2 * outWidth / 2 * N;

  float tmp_u[12];  // 4 * 3
  float u[16];      // 4 * 4;
  // U[:, :, k, c] = G * filters[k, c, :, :] * G.T()
  #pragma omp parallel for private(tmp_u, u) collapse(2)
  for (int k = 0; k < K; ++k) {
    for (int c = 0; c < C; ++c) {
      float *filters_ptr = filter + (k * C + c) * sizeF;
      sgemm433(&G[0][0], filters_ptr, tmp_u);
      sgemm434(tmp_u, &G_T[0][0], u);

      for (int xi = 0; xi < 4; ++xi)
        for (int nu = 0; nu < 4; ++nu)
          U[((xi * 4 + nu) * K + k) * C + c] = u[xi * 4 + nu];
      /*
      for (int xi = 0; xi < 4; ++xi){
        U[((xi * 4) * K + k) * C + c] = u[xi * 4];
        U[((xi * 4 + 1) * K + k) * C + c] = u[xi * 4 + 1];
        U[((xi * 4 + 2) * K + k) * C + c] = u[xi * 4 + 2];
        U[((xi * 4 + 3) * K + k) * C + c] = u[xi * 4 + 3];
      }*/

    }
  }
  // V[:, :, c, p] = B_T * image[c, b, :, :] * B
  float tmp_v[16];
  float d[16];  // d: [4 * 4];
  float v[16];  // v: [4 * 4];

  #pragma omp parallel for private(tmp_v, d, v) collapse(4)
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c) {
      for (int y = 0; y < outHeight / 2; ++y) {
        for (int x = 0; x < outWidth / 2; ++x) {
          // Generate d_cb

          for (int iy = 0; iy < 4; ++iy)
            for (int ix = 0; ix < 4; ++ix)
              d[iy * 4 + ix] = image[(n * C + c) * sizeI +
                                     (y * 2 + iy) * inWidth + (x * 2 + ix)];
          /*
          register __m128 temp0 = _mm_loadu_ps(image + (n * C + c) * sizeI + (y * 2) * inWidth + (x * 2));
          register __m128 temp1 = _mm_loadu_ps(image + (n * C + c) * sizeI + (y * 2 + 1) * inWidth + (x * 2));
          register __m128 temp2 = _mm_loadu_ps(image + (n * C + c) * sizeI + (y * 2 + 2) * inWidth + (x * 2));
          register __m128 temp3 = _mm_loadu_ps(image + (n * C + c) * sizeI + (y * 2 + 3) * inWidth + (x * 2));

          _mm_storeu_ps(d,    temp0);
          _mm_storeu_ps(d+4,  temp1);
          _mm_storeu_ps(d+8,  temp2);
          _mm_storeu_ps(d+12, temp3);
          */
          sgemm444(&B_T[0][0], d, tmp_v);
          sgemm444(tmp_v, &B[0][0], v);

          int b = ((n * outHeight / 2) + y) * outWidth / 2 + x;

          for (int xi = 0; xi < 4; ++xi)
            for (int nu = 0; nu < 4; ++nu)
              V[((xi * 4 + nu) * C + c) * P + b] = v[xi * 4 + nu];
          /*
          for (int xi = 0; xi < 4; ++xi){
            V[((xi * 4) * C + c) * P + b] = v[xi * 4];
            V[((xi * 4 + 1) * C + c) * P + b] = v[xi * 4 + 1];
            V[((xi * 4 + 2) * C + c) * P + b] = v[xi * 4 + 2];
            V[((xi * 4 + 3) * C + c) * P + b] = v[xi * 4 + 3];
          }
          */
        }
      }
    }

  // M[xi, nu, :, :] = U[xi, nu, :, :] * V[xi, nu, :, :]
  #pragma omp parallel for collapse(2)
  for (int xi = 0; xi < 4; ++xi) {
    for (int nu = 0; nu < 4; ++nu) {
      float *M_ptr = M + (xi * 4 + nu) * K * P;
      float *U_ptr = U + (xi * 4 + nu) * K * C;
      float *V_ptr = V + (xi * 4 + nu) * C * P;
      sgemm_with_kernel(U_ptr, V_ptr, M_ptr, K, C, P);
      //cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, P, C, 1.0, U_ptr, C, V_ptr, P, 0.0, M_ptr, P);
    }
  }

  // Y = A_T * m * A
  float mm[16];       // 4 * 4
  float temp_out[4];  // 2 * 2
  float tmp_m[8];     // 2 * 4
  #pragma omp parallel for private(mm, temp_out, tmp_m) collapse(4)
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) {
      for (int y = 0; y < outHeight / 2; ++y) {
        for (int x = 0; x < outWidth / 2; ++x) {
          int b = (n * outHeight / 2 + y) * outWidth / 2 + x;

          for (int xi = 0; xi < 4; ++xi) {
            for (int nu = 0; nu < 4; ++nu) {
              mm[xi * 4 + nu] = M[((xi * 4 + nu) * K + k) * P + b];
            }
          }

          /*
          for (int xi = 0; xi < 4; ++xi) {
            mm[xi * 4] = M[((xi * 4) * K + k) * P + b];
            mm[xi * 4 + 1] = M[((xi * 4 + 1) * K + k) * P + b];
            mm[xi * 4 + 2] = M[((xi * 4 + 2) * K + k) * P + b];
            mm[xi * 4 + 3] = M[((xi * 4 + 3) * K + k) * P + b];
          }*/
          sgemm244(&A_T[0][0], mm, tmp_m);
          sgemm242(tmp_m, &A[0][0], temp_out);

          for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
              out[((n * K + k) * outHeight + y * 2 + i) * outWidth + x * 2 +
                  j] = temp_out[i * 2 + j];

          /*
          out[((n * K + k) * outHeight + y * 2) * outWidth + x * 2] = temp_out[0];
          out[((n * K + k) * outHeight + y * 2) * outWidth + x * 2 + 1] = temp_out[1];
          out[((n * K + k) * outHeight + y * 2 + 1) * outWidth + x * 2] = temp_out[2];
          out[((n * K + k) * outHeight + y * 2 + 1) * outWidth + x * 2 + 1] = temp_out[3];
          */
        }
      }
    }
}
