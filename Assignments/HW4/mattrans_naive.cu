#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>


#define BLOCKSIZE 32

__global__ void transposeNaive(float *A, float *B, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col

    if (i < n && j < n) {
        B[j * n + i] = A[i * n + j];
    }
}


void initMatrix(float *mat, int n) {
    for (int i = 0; i < n*n; ++i)
        mat[i] = static_cast<float>(i);
}

void checkResult(float *A, float *B, int n) {
    bool correct = true;
    for (int i = 0; i < n && correct; ++i)
        for (int j = 0; j < n && correct; ++j)
            if (A[i*n + j] != B[j*n + i])
                correct = false;
    if (correct)
        std::cout << "Transpose correct.\n";
    else
        std::cout << "Transpose incorrect!\n";
}

int main() {
    int n = 4096; // or 2048

    size_t bytes = n * n * sizeof(float);
    float *h_A = (float*) malloc(bytes);
    float *h_B = (float*) malloc(bytes);

    float *d_A
    cudaMalloc(&d_A, bytes);


    initMatrix(h_A, n);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);


    dim3 blockDims(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDims((n + BLOCKSIZE - 1) / BLOCKSIZE, (n + BLOCKSIZE - 1) / BLOCKSIZE);

    // -------- Naive version --------
    cudaMemset(d_B, 0, bytes);

    transposeNaive<<<gridDims, blockDims>>>(d_A, d_B, n);

    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
    checkResult(h_A, h_B, n);

    // free(h_A);
    // free(h_B);
    // cudaFree(d_A);
    // cudaFree(d_B);

    return 0;
}