#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <time.h>

#define M 10
#define N 2048

int main()
{
  gsl_matrix_float *I = gsl_matrix_float_alloc(N, N);
  gsl_matrix_float *A = gsl_matrix_float_alloc(N, N);
  for (long i = 0; i < N; ++i)
    for (long j = 0; j < N; ++j)
    {
      if (i == j)
        gsl_matrix_float_set(I, i, j, 1);
      gsl_matrix_float_set(A, i, j, (float) rand() / rand());
    }
  gsl_matrix_float* B = gsl_matrix_float_alloc(N, N);
  gsl_matrix_float* in_A = gsl_matrix_float_alloc(N, N);
  gsl_matrix_float* R = gsl_matrix_float_alloc(N, N);
  gsl_matrix_float* AT = gsl_matrix_float_alloc(N, N);
  gsl_matrix_float_memcpy(AT, A);
  gsl_matrix_float_transpose(AT);
  clock_t start;
  double used_time;
  start = clock(); 
  float AA = gsl_matrix_float_norm1(A) *  gsl_matrix_float_norm1(AT);
  gsl_blas_sgemm(CblasTrans, CblasNoTrans, 1 / AA, A, I, 0, B);
  gsl_matrix_float_memcpy(R, I);
  gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, -1, B, A, 1, R);
  gsl_matrix_float_memcpy(in_A, B);
  for(u_char i = 0; i < M - 1; ++i){
    gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1, R, B, 0, B);
    gsl_blas_sgemm(CblasNoTrans, CblasNoTrans, 1, B, I, 1, A);
   }  
  used_time = (double)(clock() - start) / CLOCKS_PER_SEC;
  printf("%d: %f seconds\n", N, used_time);
  return 0;
}
