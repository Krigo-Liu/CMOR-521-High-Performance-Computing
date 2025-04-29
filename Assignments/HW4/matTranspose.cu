#include <cuda.h>
#include <iostream>
#include <cstdio>

#define BLOCKSIZE 32 // or 16, will be passed at compile time


__global__ void transposeNaive(float *A, float *B, int N) {
    int block_row = blockIdx.y * BLOCKSIZE;
    int block_col = blockIdx.x * BLOCKSIZE;
    
    int tx = threadIdx.x; // thread id within the block

    for (int i = 0; i < BLOCKSIZE && (block_row + i) < N; i++) {
        int row = block_row + i;
        int col = block_col + tx;
        if (col < N) {
            B[col * N + row] = A[row * N + col];
        }
    }
}

__global__ void transposeShared(float *A, float *B, int N) {
    __shared__ float tile[BLOCKSIZE][BLOCKSIZE+1];

    int block_row = blockIdx.y * BLOCKSIZE;
    int block_col = blockIdx.x * BLOCKSIZE;

    int tx = threadIdx.x;

    // Read A into shared memory
    for (int i = 0; i < BLOCKSIZE && (block_row + i) < N; i++) {
        int row = block_row + i;
        int col = block_col + tx;
        if (col < N) {
            tile[i][tx] = A[row * N + col];
        }
    }

    __syncthreads();

    // Write transposed data from shared memory to B
    for (int i = 0; i < BLOCKSIZE && (block_col + i) < N; i++) {
        int row = block_col + i;
        int col = block_row + tx;
        if (col < N) {
            B[row * N + col] = tile[tx][i];
        }
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
    //dim3 blockDims(BLOCKSIZE, BLOCKSIZE);
    dim3 blockDims(BLOCKSIZE);


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
