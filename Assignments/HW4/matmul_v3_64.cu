#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 64

__global__ void matmul(int N, const float *A, const float *B, float *C) {

    // Create shared memory titles for A and B
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // Calculate thread row and column inside the matrix
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    float tmp = 0.0f;

    // Loop over all tiles required
    for (int bkIdx = 0; bkIdx < N; bkIdx += BLOCKSIZE) {
        // Load one element of A and B into shared memory, if within bounds
        if (i < N && (bkIdx + threadIdx.x) < N)
            As[threadIdx.y * BLOCKSIZE + threadIdx.x] = A[i * N + (bkIdx + threadIdx.x)];
        else
            As[threadIdx.y * BLOCKSIZE + threadIdx.x] = 0.0f;

        if (j < N && (bkIdx + threadIdx.y) < N)
            Bs[threadIdx.y * BLOCKSIZE + threadIdx.x] = B[(bkIdx + threadIdx.y) * N + j];
        else
            Bs[threadIdx.y * BLOCKSIZE + threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < BLOCKSIZE; ++k) {
            tmp += As[threadIdx.y * BLOCKSIZE + k] * Bs[k * BLOCKSIZE + threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result back to C
    if (j < N && i < N) {
        C[j * N + i] = tmp;
    }
}


int main(int argc, char * argv[]){

    int N = 4096;
    if (argc > 1) N = atoi(argv[1]);

    float * A = new float[N * N];
    float * B = new float[N * N];
    float * C = new float[N * N];
  
    for (int i = 0; i < N * N; ++i){
      A[i] = 0.f;
      B[i] = 0.f;
      C[i] = 0.f;
    }
    for (int i = 0; i < N; ++i){
      A[i + i * N] = 1.f; // identity
      B[i + i * N] = 1.f; // identity
    }
  
    // allocate memory and copy to the GPU
    float * d_A;
    float * d_B;
    float * d_C;
    int size = N * N * sizeof(float);
    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_B, size);
    cudaMalloc((void **) &d_C, size);
    
    // copy memory over to the GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, size, cudaMemcpyHostToDevice);
  
    // Next largest multiple of blockSize
    int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE; 
    printf("N = %d, blocksize = %d, numBlocks * blockSize = %d\n", N, BLOCKSIZE, numBlocks * BLOCKSIZE);
    dim3 gridDims(numBlocks, numBlocks);
    dim3 blockDims(BLOCKSIZE, BLOCKSIZE);
    matmul <<< gridDims, blockDims >>> (N, d_A, d_B, d_C);

    // copy memory back to the CPU
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    
    float error = 0.f;
    for (int i = 0; i < N; ++i){
        for (int j = 0; j < N; ++j){
        //      printf("C[%d,%d] = %f\n", i, j, C[j + i * N]);
        float Cij = 0.f;
        if (i==j){
        Cij = 1.f;
        }
        float diff = C[j + i * N] - Cij;
        error += fabs(diff);
        }
    }
    printf("error = %f\n", error);

    return 0;
}