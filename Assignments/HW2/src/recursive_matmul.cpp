#include "../include/recursive_matmul.hpp"

// Helper function: compute the smallest power of two greater than or equal to x.
int nextPowerOfTwo(int x) {
    int power = 1;
    while (power < x) {
        power *= 2;
    }
    return power;
}

// Initialize an n x n matrix with the given constant value.
void initializeMatrix(double* M, int n, double value) {
    for (int i = 0; i < n * n; i++) {
        M[i] = value;
    }
}

// Recursive matrix multiplication using tasks.
// fullSize: dimension (stride) of the full matrix (used for indexing).
// currentSize: current submatrix dimension.
void recursiveMatMul(const int fullSize, const int currentSize, double* C, double* A, double* B) {
    if (currentSize <= MIN_RECURSIVE_SIZE) {
        // Base case: perform standard triple-nested loop multiplication.
        for (int i = 0; i < currentSize; ++i) {
            for (int j = 0; j < currentSize; ++j) {
                double sum = C[i * fullSize + j];
                for (int k = 0; k < currentSize; ++k) {
                    sum += A[i * fullSize + k] * B[k * fullSize + j];
                }
                C[i * fullSize + j] = sum;
            }
        }
    } else {
        int half = currentSize / 2;
        // Define pointers for the four submatrices (using fullSize as stride).
        double *A11 = A;
        double *A12 = A + half;
        double *A21 = A + half * fullSize;
        double *A22 = A + half * fullSize + half;

        double *B11 = B;
        double *B12 = B + half;
        double *B21 = B + half * fullSize;
        double *B22 = B + half * fullSize + half;

        double *C11 = C;
        double *C12 = C + half;
        double *C21 = C + half * fullSize;
        double *C22 = C + half * fullSize + half;

        // Each quadrant of C gets two contributions.
        #pragma omp task if(currentSize > MIN_RECURSIVE_SIZE * 2) firstprivate(A11, A12, B11, B21, C11, half, fullSize)
        {
            recursiveMatMul(fullSize, half, C11, A11, B11);
            recursiveMatMul(fullSize, half, C11, A12, B21);
        }
        #pragma omp task if(currentSize > MIN_RECURSIVE_SIZE * 2) firstprivate(A11, A12, B12, B22, C12, half, fullSize)
        {
            recursiveMatMul(fullSize, half, C12, A11, B12);
            recursiveMatMul(fullSize, half, C12, A12, B22);
        }
        #pragma omp task if(currentSize > MIN_RECURSIVE_SIZE * 2) firstprivate(A21, A22, B11, B21, C21, half, fullSize)
        {
            recursiveMatMul(fullSize, half, C21, A21, B11);
            recursiveMatMul(fullSize, half, C21, A22, B21);
        }
        #pragma omp task if(currentSize > MIN_RECURSIVE_SIZE * 2) firstprivate(A21, A22, B12, B22, C22, half, fullSize)
        {
            recursiveMatMul(fullSize, half, C22, A21, B12);
            recursiveMatMul(fullSize, half, C22, A22, B22);
        }
        #pragma omp taskwait
    }
}

// Strong scaling experiment: fixed matrix size while varying the number of threads.
void runStrongScalingExperiment() {
    vector<Result> results;
    int threadCounts[] = {1, 2, 4, 8, 16};
    int numTests = sizeof(threadCounts) / sizeof(threadCounts[0]);
    double baselineTime = 0.0;

    cout << "Strong Scaling Experiment (Matrix Size = " << STRONG_MATRIX_SIZE << " x " << STRONG_MATRIX_SIZE << ")\n";

    for (int i = 0; i < numTests; i++) {
        int numThreads = threadCounts[i];
        omp_set_num_threads(numThreads);
        int n = STRONG_MATRIX_SIZE;

        // Allocate matrices.
        double* A = new double[n * n];
        double* B = new double[n * n];
        double* C = new double[n * n];

        // Initialize A and B to 1.0 and C to 0.0.
        initializeMatrix(A, n, 1.0);
        initializeMatrix(B, n, 1.0);
        initializeMatrix(C, n, 0.0);

        double start = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            {
                recursiveMatMul(n, n, C, A, B);
            }
        }
        double end = omp_get_wtime();
        double runtime = end - start;
        if (numThreads == 1) {
            baselineTime = runtime;
        }
        // For strong scaling, the ideal speedup is proportional to the number of threads.
        double efficiency = (baselineTime / runtime) / numThreads;

        results.push_back({"Recursive MatMul", numThreads, n, runtime, efficiency});

        delete[] A;
        delete[] B;
        delete[] C;
    }

    cout << "\nRecursive MatMul Results (Strong Scaling):\n";
    cout << std::left
         << std::setw(20) << "Version"
         << std::setw(20) << "Threads"
         << std::setw(20) << "Matrix Size"
         << std::setw(20) << "Runtime"
         << std::setw(20) << "Efficiency"
         << std::endl; 
    cout << fixed << setprecision(5);
    for (const auto &res : results) {
        cout << left
             << setw(20) << res.version 
             << setw(20) << res.numThreads 
             << setw(20) << res.matrixSize
             << setw(20) << res.runtime 
             << setw(20) << res.efficiency 
             << endl;
    }
    std::cout << std::string(20 + 10 + 12 + 12 + 10, '-') << std::endl;
}

// Weak scaling experiment: matrix size increases with thread count so that work per thread remains roughly constant.
void runWeakScalingExperiment() {
    vector<Result> results;
    int threadCounts[] = {1, 2, 4, 8, 16};
    int numTests = sizeof(threadCounts) / sizeof(threadCounts[0]);
    double baselineTime = 0.0;

    cout << "Weak Scaling Experiment (Base Matrix Size = " << WEAK_BASE_MATRIX_SIZE 
         << " x " << WEAK_BASE_MATRIX_SIZE << " for 1 thread)\n";

    for (int i = 0; i < numTests; i++) {
        int numThreads = threadCounts[i];
        omp_set_num_threads(numThreads);

        // Scale the matrix size: total work ∝ n^3 should increase by the factor of numThreads.
        // int candidateSize = static_cast<int>(ceil(WEAK_BASE_MATRIX_SIZE * cbrt(static_cast<double>(numThreads))));
        // int n = nextPowerOfTwo(candidateSize);
        int n = static_cast<int>(std::ceil(WEAK_BASE_MATRIX_SIZE * static_cast<double>(numThreads)));


        double* A = new double[n * n];
        double* B = new double[n * n];
        double* C = new double[n * n];

        initializeMatrix(A, n, 1.0);
        initializeMatrix(B, n, 1.0);
        initializeMatrix(C, n, 0.0);

        double start = omp_get_wtime();
        #pragma omp parallel
        {
            #pragma omp single
            {
                recursiveMatMul(n, n, C, A, B);
            }
        }
        double end = omp_get_wtime();
        double runtime = end - start;
        if (numThreads == 1) {
            baselineTime = runtime;
        }
        // For weak scaling, ideally the runtime remains constant.
        double efficiency = baselineTime / runtime;

        results.push_back({"Recursive MatMul", numThreads, n, runtime, efficiency});

        delete[] A;
        delete[] B;
        delete[] C;
    }

    cout << "\nRecursive MatMul Results (Weak Scaling):\n";
    cout << std::left
         << std::setw(20) << "Version"
         << std::setw(20) << "Threads"
         << std::setw(20) << "Matrix Size"
         << std::setw(20) << "Runtime"
         << std::setw(20) << "Efficiency"
         << std::endl; 
    cout << fixed << setprecision(5);
    for (const auto &res : results) {
        cout << left
             << setw(20) << res.version 
             << setw(20) << res.numThreads 
             << setw(20) << res.matrixSize
             << setw(20) << res.runtime 
             << setw(20) << res.efficiency 
             << endl;
    }

}

