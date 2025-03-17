#ifndef CACHE_BLOCK_MATMUL
#define CACHE_BLOCK_MATMUL

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <omp.h>

using namespace std;

// Structure to hold the result for each test.
struct Result {
    string version;
    int numThreads;
    int matrixSize;  // For weak scaling experiments.
    double runtime;
    double efficiency;
};

// Global parameters for matrix multiplication.
extern int STRONG_N;  // Fixed matrix dimension for strong scaling
extern int WEAK_BASE_N;   // Base matrix dimension for 1 thread in weak scaling
extern int BLOCK_SIZE;    // Block size for cache blocking

void init_matrix(double* M, int n, double value);

void cache_block_matmul_standard(double *A, double *B, double *C, int n);

void cache_block_matmul_collapse(double *A, double *B, double *C, int n);

void cache_block_matmul_collapse_3(double *A, double *B, double *C, int n);

void run_strong_scaling_experiment();

void run_weak_scaling_experiment();

#endif // CACHE_BLOCK_MATMUL