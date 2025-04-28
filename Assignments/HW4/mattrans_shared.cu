#include <iostream>
#include <cstdio>
#include <cuda_runtime.h>


#define BLOCKSIZE 32


__global__ void transposeShared(float *A, float *B, int n) {
    __shared__ float tile[BLOCKSIZE][BLOCKSIZE+1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * BLOCKSIZE + threadIdx.x;
    int y = blockIdx.y * BLOCKSIZE + threadIdx.y;

    // Read into shared memory
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = A[y * n + x];
    }

    __syncthreads();

    // Write out transposed
    x = blockIdx.y * BLOCKSIZE + threadIdx.x;
    y = blockIdx.x * BLOCKSIZE + threadIdx.y;

    if (x < n && y < n) {
        B[y * n + x] = tile[threadIdx.x][threadIdx.y];
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

    float *d_A, *d_B;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);


    initMatrix(h_A, n);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);


    dim3 blockDims(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDims((n + BLOCKSIZE - 1) / BLOCKSIZE, (n + BLOCKSIZE - 1) / BLOCKSIZE);


    // -------- Shared memory version --------
    cudaMemset(d_B, 0, bytes);

    transposeShared<<<gridDims, blockDims>>>(d_A, d_B, n);


    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
    checkResult(h_A, h_B, n);

    // free(h_A);
    // free(h_B);
    // cudaFree(d_A);
    // cudaFree(d_B);

    return 0;
}