#include <stdio.h>
#include <math.h>

#define BLOCKSIZE 256

__global__ void partial_reduction_v1(const int N, float *x_reduced, 
                                     const float *x){
  // This is a shared memory array for each thread block
  // All threads within the same block can read/write to the same shared memory array.
  // Threads in different blocks have completely independent shared memory spaces.
  __shared__ float s_x[BLOCKSIZE];
  // blickDim means how many threads in a thread block
  // blockIdx means which block is it
  // threadIdx.x means which threads it is  
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;
  
  s_x[tid] = 0.f;
  if (i < N){
    s_x[tid] = x[i];
  }

  for (int s = 1; s < blockDim.x; s *= 2){
    if (tid % (2 * s) == 0){
        s_x[tid] += s_x[tid + s];
    }
    __syncthreads(); 
  }

  // write out once we're done reducing each block
  if (tid==0){
    x_reduced[blockIdx.x] = s_x[0];
  }
}

__global__ void partial_reduction_v2(const int N, float *x_reduced, 
                                     const float *x){
  
  __shared__ float s_x[BLOCKSIZE];

  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;
  
  s_x[tid] = 0.f;
  if (i < N){
    s_x[tid] = x[i];
  }
  __syncthreads(); 

  for (int s = 1; s < blockDim.x; s *= 2){
    int index = 2 * s * tid;
    if (index < blockDim.x){
        s_x[index] += s_x[index + s];
    }
    __syncthreads(); 
  }

  // write out once we're done reducing each block
  if (tid==0){
    x_reduced[blockIdx.x] = s_x[0];
  }
}

__global__ void partial_reduction_v3(const int N, float *x_reduced, 
                                     const float *x){
  
  __shared__ float s_x[BLOCKSIZE];

  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;
  
  // coalesced reads in
  s_x[tid] = 0.f;
  if (i < N){
    s_x[tid] = x[i];
  }

  // number of "live" threads per block
  int alive = blockDim.x;
  
  while (alive > 1){
    __syncthreads(); 
    alive /= 2; // update the number of live threads    
    if (tid < alive){
      s_x[tid] += s_x[tid + alive];
    }
  }

  // write out once we're done reducing each block
  if (tid==0){
    x_reduced[blockIdx.x] = s_x[0];
  }
}

__global__ void partial_reduction_v4(const int N, float *x_reduced, 
                                     const float *x){
  
  __shared__ float s_x[BLOCKSIZE];

  const int i = (2 * blockDim.x) * blockIdx.x + threadIdx.x;
  const int tid = threadIdx.x;
  
  // coalesced reads
  s_x[tid] = 0.f;
  if (i < N){
    s_x[tid] = x[i] + x[i + blockDim.x];
  }

  // number of "live" threads per block
  int alive = blockDim.x;
  
  while (alive > 1){
    __syncthreads();
    alive /= 2; 
    if (tid < alive){
      s_x[tid] += s_x[tid + alive];
    }
  }

  // write out once we're done reducing each block
  if (tid==0){
    x_reduced[blockIdx.x] = s_x[0];
  }
}

void check_reduction_error(int numBlocks, float* x_reduced, float target){
  float sum_x = 0.f;
  for (int i = 0; i < numBlocks; ++i){
    sum_x += x_reduced[i];
  }
  printf("error = %f\n", fabs(sum_x - target));
}

int main(int argc, char * argv[]){

  int N = 4096;
  if (argc > 1){
    N = atoi(argv[1]);
  }

  // Next largest multiple of blockSize
  int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;

  printf("N = %d, blockSize = %d, numBlocks = %d\n", N, BLOCKSIZE, numBlocks);

  float * x = new float[N];
  float * x_reduced = new float[numBlocks];  

  for (int i = 0; i < N; ++i){
    x[i] = 1.f;
  }

  // allocate memory and copy to the GPU
  float * d_x;
  float * d_x_reduced;  
  int size_x = N * sizeof(float);
  int size_x_reduced = numBlocks * sizeof(float);
  cudaMalloc((void **) &d_x, size_x);
  cudaMalloc((void **) &d_x_reduced, size_x_reduced);
  
  // copy memory over to the GPU
  cudaMemcpy(d_x, x, size_x, cudaMemcpyHostToDevice);
  cudaMemcpy(d_x_reduced, x_reduced, size_x_reduced, cudaMemcpyHostToDevice);

  // check version 1
  partial_reduction_v1 <<< numBlocks, BLOCKSIZE >>> (N, d_x_reduced, d_x);
  cudaMemcpy(x_reduced, d_x_reduced, size_x_reduced, cudaMemcpyDeviceToHost);
  float target = N;
  printf("For reduction version 1: ");
  check_reduction_error(numBlocks, x_reduced, target);

  // check version 2
  partial_reduction_v2 <<< numBlocks, BLOCKSIZE >>> (N, d_x_reduced, d_x);
  cudaMemcpy(x_reduced, d_x_reduced, size_x_reduced, cudaMemcpyDeviceToHost);
  printf("For reduction version 2: ");
  check_reduction_error(numBlocks, x_reduced, target);

  // check version 3
  partial_reduction_v3 <<< numBlocks, BLOCKSIZE >>> (N, d_x_reduced, d_x);
  cudaMemcpy(x_reduced, d_x_reduced, size_x_reduced, cudaMemcpyDeviceToHost);
  printf("For reduction version 3: ");
  check_reduction_error(numBlocks, x_reduced, target);

  // check version 4. 
  // note that this version is run with half the blocks
  int numBlocksHalf = (numBlocks + 1) / 2;
  partial_reduction_v4 <<< numBlocksHalf, BLOCKSIZE >>> (N, d_x_reduced, d_x);
  cudaMemcpy(x_reduced, d_x_reduced, size_x_reduced, cudaMemcpyDeviceToHost);
  printf("For reduction version 4: ");
  check_reduction_error(numBlocksHalf, x_reduced, target);

  #if 1
  float time;
  float min_time = 1e6;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // version 1
  for (int i = 0; i < 10; ++i)
  {
    cudaEventRecord(start, 0);

    partial_reduction_v1 <<< numBlocks, BLOCKSIZE >>> (N, d_x_reduced, d_x);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    if (time < min_time)
      min_time = time;
  }
  printf("Time to run v1 kernel: %6.2f ms.\n", min_time);

  // version 2
  for (int i = 0; i < 10; ++i)
  {
    cudaEventRecord(start, 0);

    partial_reduction_v2 <<< numBlocks, BLOCKSIZE >>> (N, d_x_reduced, d_x);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    if (time < min_time)
      min_time = time;
  }
  printf("Time to run v2 kernel: %6.2f ms.\n", min_time);

  // version 3
  for (int i = 0; i < 10; ++i)
  {
    cudaEventRecord(start, 0);

    partial_reduction_v3 <<< numBlocks, BLOCKSIZE >>> (N, d_x_reduced, d_x);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    if (time < min_time)
      min_time = time;
  }
  printf("Time to run v3 kernel: %6.2f ms.\n", min_time);

  // version 4
  for (int i = 0; i < 10; ++i)
  {
    cudaEventRecord(start, 0);

    partial_reduction_v4 <<< numBlocksHalf, BLOCKSIZE >>> (N, d_x_reduced, d_x);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    if (time < min_time)
      min_time = time;
  }
  printf("Time to run v4 kernel: %6.2f ms.\n", min_time);

#endif

  return 0;
}
