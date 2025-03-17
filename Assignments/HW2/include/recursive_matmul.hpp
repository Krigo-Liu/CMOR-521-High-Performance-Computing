#ifndef RECURSIVE_MATMUL
#define RECURSIVE_MATMUL

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


// Structure to store each test result.
struct Result {
    string version;
    int numThreads;
    int matrixSize;  // For weak scaling experiments.
    double runtime;
    double efficiency;
};

// For strong scaling, the matrix size remains fixed (must be a power of 2).
extern int STRONG_MATRIX_SIZE; 
// For weak scaling, the 1-thread matrix size is the base, then scaled by cube-root of thread count.
extern int WEAK_BASE_MATRIX_SIZE;
// Threshold for switching to base-case multiplication in recursive calls (must be a power of 2).
extern int MIN_RECURSIVE_SIZE;


// Helper function: compute the smallest power of two greater than or equal to x.
int nextPowerOfTwo(int x);

// Initialize an n x n matrix with the given constant value.
void initializeMatrix(double* M, int n, double value);

// Recursive matrix multiplication using tasks.
void recursiveMatMul(const int fullSize, const int currentSize, double* C, double* A, double* B);

// Strong scaling experiment: fixed matrix size while varying the number of threads.
void runStrongScalingExperiment();

// Weak scaling experiment: matrix size increases with thread count so that work per thread remains roughly constant.
void runWeakScalingExperiment();

#endif // RECURSIVE_MATMUL