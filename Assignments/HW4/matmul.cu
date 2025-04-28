#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 64

//------------------------------------------------------------
// v1: naive version (original i=row, j=col)
//------------------------------------------------------------
__global__ void matmul_v1(int N, const float *A, const float *B, float *C) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        float val = 0.f;
        for (int k = 0; k < N; ++k) {
            val += A[k + i * N] * B[j + k * N];
        }
        C[j + i * N] = val;
    }
}

//------------------------------------------------------------
// v2: exchange i and j (j=row, i=col)
//------------------------------------------------------------
__global__ void matmul_v2(int N, const float *A, const float *B, float *C) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < N && j < N) {
        float val = 0.f;
        for (int k = 0; k < N; ++k) {
            val += A[k + i * N] * B[j + k * N];
        }
        C[j + i * N] = val; 
    }
}

//------------------------------------------------------------
// v3: shared memory tiled version
//------------------------------------------------------------
__global__ void matmul_v3(int N, const float *A, const float *B, float *C) {
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.0f;

    for (int bkIdx = 0; bkIdx < N; bkIdx += BLOCKSIZE) {
        if (i < N && (bkIdx + threadIdx.x) < N)
            As[threadIdx.y * BLOCKSIZE + threadIdx.x] = A[i * N + (bkIdx + threadIdx.x)];
        else
            As[threadIdx.y * BLOCKSIZE + threadIdx.x] = 0.0f;

        if (j < N && (bkIdx + threadIdx.y) < N)
            Bs[threadIdx.y * BLOCKSIZE + threadIdx.x] = B[(bkIdx + threadIdx.y) * N + j];
        else
            Bs[threadIdx.y * BLOCKSIZE + threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; ++k) {
            tmp += As[threadIdx.y * BLOCKSIZE + k] * Bs[k * BLOCKSIZE + threadIdx.x];
        }

        __syncthreads();
    }

    if (j < N && i < N) {
        C[j * N + i] = tmp;
    }
}

//------------------------------------------------------------
// main function
//------------------------------------------------------------
int main(int argc, char *argv[]) {
    int N = 4096;
    int version = 1; // default use v1

    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) version = atoi(argv[2]);



    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];

    for (int i = 0; i < N * N; ++i) {
        A[i] = 0.f;
        B[i] = 0.f;
        C[i] = 0.f;
    }
    for (int i = 0; i < N; ++i) {
        A[i + i * N] = 1.f;
        B[i + i * N] = 1.f;
    }

    float *d_A, *d_B, *d_C;
    int size = N * N * sizeof(float);
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    printf("N = %d, numBlocks * blockSize = %d, using matmul_v%d\n", N, numBlocks * BLOCKSIZE, version);
    dim3 gridDims(numBlocks, numBlocks);
    dim3 blockDims(BLOCKSIZE, BLOCKSIZE);

    if (version == 1) {
        matmul_v1<<<gridDims, blockDims>>>(N, d_A, d_B, d_C);
    }
    else if (version == 2) {
        matmul_v2<<<gridDims, blockDims>>>(N, d_A, d_B, d_C);
    }
    else if (version == 3) {
        matmul_v3<<<gridDims, blockDims>>>(N, d_A, d_B, d_C);
    }
    else {
        printf("Error: unknown version %d\n", version);
        return -1;
    }

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    float error = 0.f;
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
            float expected = (i == j) ? 1.f : 0.f;
            error += fabs(C[j + i * N] - expected);
        }
    }
    printf("Error = %f\n", error);

    delete[] A;
    delete[] B;
    delete[] C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
