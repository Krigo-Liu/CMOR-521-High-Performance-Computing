#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

__global__ void add_v1(int N, const float *A, float *B){
  
  int index = blockIdx.x * blockDim.x + threadIdx.x;  
  int i = index / N;
  int j = index % N;
  if (i < N && j < N){
    A[i + j * N] += B[i + j * N];
  }
}

__global__ void add_v2(int N, const float *A, float *B){
  
    int i = blockIdx.x * blockDim.x + threadIdx.x;  
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N){
      A[i + j * N] += B[i + j * N];
    }
  }

int main(void){

  int N = 1024;
  float * A = new float[N * N];
  float * B = new float[N * N];

  for (int i = 0; i < N * N; ++i){
    A[i] = 1.f - i;
    B[i] = (float) i;
  }

  int size = N * N * sizeof(float);

  // allocate memory and copy to the GPU
  float * d_A;
  float * d_B;
  cudaMalloc((void **) &d_A, size);
  cudaMalloc((void **) &d_B, size);
  
  // copy memory over to the GPU
  cudaMemcpy(d_A, x, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, y, size, cudaMemcpyHostToDevice);

  // call the add kernel
  int blockSize = 2048;
  int gridSize = (N + blockSize - 1) / blockSize;
  add_v1<<<gridSize, blockSize>>>(N, d_A, d_B);

//   dim3 blockSize(128, 128);
//   dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
//                 (N + blockSize.y - 1) / blockSize.y);
//   add_v2<<<gridSize, blockSize>>>(N, d_A, d_B);

  // copy memory back to the CPU
  cudaMemcpy(A, d_B, size, cudaMemcpyDeviceToHost);
  
  float error = 0.f;
  for (int i = 0; i < N; ++i){
    error += fabs(A[i] - 1.f);
  }
  printf("error = %f\n", error);

  return 0;
}
