#include <cuda.h>
#include <iostream>
#include <cstdio>

#define BLOCKSIZE 32 // or 64, will be passed at compile time
__global__ void transposeNaive(float *A, float *B, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; // row
    int j = blockIdx.x * blockDim.x + threadIdx.x; // col

    if (i < N && j < N) {
        B[j * N + i] = A[i * N + j];
    }
}

__global__ void transposeShared(float *A, float *B, int N) {
    __shared__ float tile[BLOCKSIZE][BLOCKSIZE+1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * BLOCKSIZE + threadIdx.x;
    int y = blockIdx.y * BLOCKSIZE + threadIdx.y;

    // Read into shared memory
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }

    __syncthreads();

    // Write out transposed
    x = blockIdx.y * BLOCKSIZE + threadIdx.x;
    y = blockIdx.x * BLOCKSIZE + threadIdx.y;

    if (x < N && y < N) {
        B[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}


void initMatrix(float *mat, int N) {
    for (int i = 0; i < N*N; ++i)
        mat[i] = static_cast<float>(i);
}

void checkResult(float *A, float *B, int N) {
    bool correct = true;
    for (int i = 0; i < N && correct; ++i)
        for (int j = 0; j < N && correct; ++j)
            if (A[i*N + j] != B[j*N + i])
                correct = false;
    if (correct)
        std::cout << "Transpose correct.\n";
    else
        std::cout << "Transpose incorrect!\n";
}

int main(int argc, char *argv[]) {
    int N = 4096; // or 2048
    
    if (argc > 1) N = atoi(argv[1]);

    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*) malloc(bytes);
    float *h_B = (float*) malloc(bytes);

    float *d_A, *d_B;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);

    initMatrix(h_A, N);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    printf("N = %d, numBlocks * blockSize = %d\n", N, numBlocks * BLOCKSIZE);
    dim3 gridDims(numBlocks, numBlocks);
    dim3 blockDims(BLOCKSIZE, BLOCKSIZE);


    // -------- Naive version --------
    cudaMemset(d_B, 0, bytes);

    transposeNaive<<<gridDims, blockDims>>>(d_A, d_B, N);

    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
    checkResult(h_A, h_B, N);

    // -------- Shared memory version --------
    cudaMemset(d_B, 0, bytes);

    transposeShared<<<gridDims, blockDims>>>(d_A, d_B, N);

    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
    checkResult(h_A, h_B, N);

    free(h_A);
    free(h_B);
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
