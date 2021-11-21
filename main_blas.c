#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <time.h>

#define M 10
#define N 2048

int main()
{
  gsl_matrix *I = gsl_matrix_alloc(N, N);
  gsl_matrix *A = gsl_matrix_alloc(N, N);
  for (long i = 0; i < N; ++i)
    for (long j = 0; j < N; ++j)
    {
      if (i == j)
        gsl_matrix_set(I, i, j, 1);
      gsl_matrix_set(A, i, j, (float) rand() / rand());
    }
  gsl_matrix* B = gsl_matrix_alloc(N, N);
  gsl_matrix* in_A = gsl_matrix_alloc(N, N);
  gsl_matrix* R = gsl_matrix_alloc(N, N);
  gsl_matrix* AT = gsl_matrix_alloc(N, N);
  gsl_matrix_memcpy(AT, A);
  gsl_matrix_transpose(AT);
  clock_t start;
  double used_time;
  start = clock(); 
  float AA = gsl_matrix_norm1(A) *  gsl_matrix_norm1(AT);
  gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1 / AA, A, I, 0, B);
  gsl_matrix_memcpy(R, I);
  gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, -1, B, A, 1, R);
  gsl_matrix_memcpy(in_A, B);
  for(u_char i = 0; i < M - 1; ++i){
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, R, B, 0, B);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, B, I, 1, A);
   }  
  used_time = (double)(clock() - start) / CLOCKS_PER_SEC;
  printf("%d: %f seconds\n", N, used_time);
  return 0;
}
