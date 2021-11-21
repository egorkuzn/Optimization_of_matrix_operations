#include <stdio.h>
#include <xmmintrin.h>
#include <stdlib.h>
#include <time.h>

#define M 10
#define N 2048

void memExcptn(void){
    printf("Exception\n");
    exit(EXIT_FAILURE);
}

void matrix_gen(float* A, float* I, float* in_A){
    if(!A || !I || !in_A)
        memExcptn;
    for(long i = 0; i < N * N; ++i){
        A[i] = (float) rand() / rand();
        in_A[i] = 0;
        if(i / N == i % N){
            I[i] = 1;
        } else{
            I[i] = 0;
        }
    }           
}

void matrix_transpose(float* A, float* res){
    if(!res)
        memExcptn;
    for(long i = 0; i < N * N; ++i)
        res[i] = A[N * (N - 1 - i / N) + (N - 1 - i % N)];
}

int compare (const void * a, const void * b){
  return ( *(float*)a - *(float*)b );
}

float max_mul_A(float* A){ 
    float sigma_x[N] = {0};
    float sigma_y[N] = {0};
    if(!sigma_x || !sigma_y)
        memExcptn;
    for(long i = 0; i < N; ++i)
        for(long j = 0; j < N; ++j){
            sigma_x[i] += A[N * i + j];
            sigma_y[j] += A[N * i + j];
        }
    qsort(sigma_x, N, sizeof(float), compare);
    qsort(sigma_y, N, sizeof(float), compare);
    float xMAX = sigma_x[N - 1];
    float yMAX = sigma_y[N - 1];
    return xMAX*yMAX; 
}

void matrix_null_init(float* A){
    for(long i = 0; i < N * N; ++i)
        A[i] = 0.0;
}

void matrix_mult_scal(float* A, float k, float* res){
    for(long i = 0; i < N; ++i)
        for(long j = 0; j < N; ++j)
            res[N * i + j] = k * A[N * i + j];        
}

void matrix_mul(float* A, float* B, float* res){
    matrix_null_init(res);
    for(long i = 0; i < N; ++i)
        for(long j = 0; j < N; ++j)
            for(long k = 0; k < N; ++k)
                res[N * i + k] += A[N * i + j] * A[N * j + k];          
}

void matrix_sub_I(float* R){
    for(long i = 0; i < N; ++i)
        for(long j = 0; j < N; ++j)
            if(j == i)
                R[N * i + j] = 1 - R[N * i + j];
            else
                R[N * i + j] = - R[N * i + j];                     
}

void matrix_sum(float* A, float* B, float* res){
    for(long i = 0; i < N; i++)
        for(long j = 0; j < N; ++j)
            res[N * i + j] = A[N * i + j] + B[N * i + j];
}


int main(int argc, char* argv[]){
    float* A = (float*)malloc(N * N * sizeof(float));
    float* I = (float*)malloc(N * N * sizeof(float)); 
    float* in_A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* R = (float*)malloc(N * N * sizeof(float));
    matrix_gen(A, I, in_A);
    clock_t start;
    start = clock(); 
    double used_time;
    matrix_transpose(A, B);
    float AA = max_mul_A(A);
    matrix_mult_scal(B, 1.0 / AA, B);
    matrix_mul(B, A, R);
    matrix_sub_I(R);
    for(long i = 0; i < M; ++i){
        matrix_sum(I, in_A, in_A);
        matrix_mul(R, I, A);
        matrix_mult_scal(A, 1.0, I);
    }
    matrix_mul(B, in_A, in_A);
    used_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    free(in_A);
    free(I);
    free(A);
    free(B);
    free(R);
    printf("%d: %f seconds\n", N, used_time);
    return 0;
} 

