#include <stdio.h>
#include <xmmintrin.h>
#include <stdlib.h>
#include <time.h>

#define M 10
#define N 2048

float xmm_vector[8];

void memExcptn(void){
    printf("Exception\n");
    exit(EXIT_FAILURE);
}

void SSE_Add(float* a, float* b, float* c){
     __asm__ volatile(
        "movups %[a], %%xmm0\n\t" // move 4 a vars to xmm0
        "movups %[b], %%xmm1\n\t" //move 4 b vars to xmm1
        "addps %%xmm1, %%xmm0\n\t" //add packs: xmm0 = xmm0 * xmm1
        "movups %%xmm0, %[c]\n\t" // load results in c
        :[c]"=m"(*c)
        :[a]"m"(*a), [b]"m"(*b)
        :"%xmm0", "%xmm1"
    );  
    a += 4;
    b += 4;
    c += 4;
     __asm__ volatile(
        "movups %[a], %%xmm0\n\t" // move 4 a vars to xmm0
        "movups %[b], %%xmm1\n\t" //move 4 b vars to xmm1
        "addps %%xmm1, %%xmm0\n\t" //mult packs: xmm0 = xmm0 * xmm1
        "movups %%xmm0, %[c]\n\t" // load results in c
        :[c]"=m"(*c)
        :[a]"m"(*a), [b]"m"(*b)
        :"%xmm0", "%xmm1"
    );  
}

void SSE_Mult(float* a, float* b, float* c){
     __asm__ volatile(
        "movups %[a], %%xmm0\n\t" // move 4 a vars to xmm0
        "movups %[b], %%xmm1\n\t" //move 4 b vars to xmm1
        "mulps %%xmm1, %%xmm0\n\t" //mult packs: xmm0 = xmm0 * xmm1
        "movups %%xmm0, %[c]\n\t" // load results in c
        :[c]"=m"(*c)
        :[a]"m"(*a), [b]"m"(*b)
        :"%xmm0", "%xmm1"
    );  
    a += 4;
    b += 4;
    c += 4;
     __asm__ volatile(
        "movups %[a], %%xmm0\n\t" // move 4 a vars to xmm0
        "movups %[b], %%xmm1\n\t" //move 4 b vars to xmm1
        "mulps %%xmm1, %%xmm0\n\t" //mult packs: xmm0 = xmm0 * xmm1
        "movups %%xmm0, %[c]\n\t" // load results in c
        :[c]"=m"(*c)
        :[a]"m"(*a), [b]"m"(*b)
        :"%xmm0", "%xmm1"
    );  
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
    xmm_vector[0] = k;
    xmm_vector[1] = k;
    xmm_vector[2] = k;
    xmm_vector[3] = k;
    xmm_vector[4] = k;
    xmm_vector[5] = k;
    xmm_vector[6] = k;
    xmm_vector[7] = k;

    for(long i = 0; i < N; ++i)
        for(long j = 0; j < N; j += 8)
            SSE_Mult(A + N * i + j, xmm_vector, res + N * i + j);
}

void matrix_mul(float* A, float* B, float* res){
    matrix_null_init(res);
    for(long i = 0; i < N; ++i)
        for(long j = 0; j < N; ++j){
            xmm_vector[0] = A[N * i + j];
            xmm_vector[1] = A[N * i + j];
            xmm_vector[2] = A[N * i + j];
            xmm_vector[3] = A[N * i + j];
            xmm_vector[4] = A[N * i + j];
            xmm_vector[5] = A[N * i + j];
            xmm_vector[6] = A[N * i + j];
            xmm_vector[7] = A[N * i + j];

            for(long k = 0; k < N; k += 8)
                SSE_Mult(xmm_vector, B + N * i + k, res + N * i + k);
        }  
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
        for(long j = 0; j < N; j += 8)
            SSE_Add(A + N * i + j, B + N * i + j, res + N * i + j);
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
