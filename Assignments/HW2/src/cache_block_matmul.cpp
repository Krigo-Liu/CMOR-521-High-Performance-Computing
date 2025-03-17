#include "../include/cache_block_matmul.hpp"

// Initialize an n x n matrix with a constant value.
void init_matrix(double* M, int n, double value) {
    for (int i = 0; i < n * n; i++) {
        M[i] = value;
    }
}


// Cache-blocked matrix multiplication using omp parallel for
// Parallelizing the outer (i-block) loop
void cache_block_matmul_standard(double *A, double *B, double *C, int n) {
    int i, j, k, ii, jj, kk;
    #pragma omp parallel for private(i, j, k, ii, jj, kk) schedule(static)
    for (i = 0; i < n; i += BLOCK_SIZE) {
        for (j = 0; j < n; j += BLOCK_SIZE) {
            for (k = 0; k < n; k += BLOCK_SIZE) {
                for (ii = i; ii < ((i + BLOCK_SIZE) > n ? n : (i + BLOCK_SIZE)); ii++) {
                    for (jj = j; jj < ((j + BLOCK_SIZE) > n ? n : (j + BLOCK_SIZE)); jj++) {
                        double sum = 0.0;
                        for (kk = k; kk < ((k + BLOCK_SIZE) > n ? n : (k + BLOCK_SIZE)); kk++) {
                            sum += A[ii * n + kk] * B[kk * n + jj];
                        }
                        C[ii * n + jj] += sum;
                    }
                }
            }
        }
    }
}

// Cache-blocked matrix multiplication using omp parallel for collapse(2)
// to flatten the i- and j- block loops
void cache_block_matmul_collapse(double *A, double *B, double *C, int n) {
    int i, j, k, ii, jj, kk;
    #pragma omp parallel for collapse(2) private(i, j, k, ii, jj, kk) schedule(static)
    for (i = 0; i < n; i += BLOCK_SIZE) {
        for (j = 0; j < n; j += BLOCK_SIZE) {
            for (k = 0; k < n; k += BLOCK_SIZE) {
                for (ii = i; ii < ((i + BLOCK_SIZE) > n ? n : (i + BLOCK_SIZE)); ii++) {
                    for (jj = j; jj < ((j + BLOCK_SIZE) > n ? n : (j + BLOCK_SIZE)); jj++) {
                        double sum = 0.0;
                        for (kk = k; kk < ((k + BLOCK_SIZE) > n ? n : (k + BLOCK_SIZE)); kk++) {
                            sum += A[ii * n + kk] * B[kk * n + jj];
                        }
                        C[ii * n + jj] += sum;
                    }
                }
            }
        }
    }
}

// Strong scaling experiment: the matrix size is fixed, and we vary the number of threads.
void run_strong_scaling_experiment() {
    vector<Result> results_std;
    vector<Result> results_col;
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    double baseline_std = 0.0, baseline_col = 0.0;
    
    cout << "Strong Scaling Experiment (Matrix Size = " << STRONG_N << " x " << STRONG_N << ")\n";
    
    for (int t = 0; t < num_tests; t++) {
        int num_threads = thread_counts[t];
        omp_set_num_threads(num_threads);
        int n = STRONG_N;
        
        // Allocate matrices.
        double* A      = new double[n * n];
        double* B      = new double[n * n];
        double* C_std  = new double[n * n];
        double* C_col  = new double[n * n];
        
        // Initialize matrices: set A and B to 1.0 and C to 0.0.
        init_matrix(A, n, 1.0);
        init_matrix(B, n, 1.0);
        init_matrix(C_std, n, 0.0);
        init_matrix(C_col, n, 0.0);
        
        double start, end;
        // Run the standard version.
        start = omp_get_wtime();
        cache_block_matmul_standard(A, B, C_std, n);
        end = omp_get_wtime();
        double runtime_std = end - start;
        
        // Run the collapse(2) version.
        start = omp_get_wtime();
        cache_block_matmul_collapse(A, B, C_col, n);
        end = omp_get_wtime();
        double runtime_col = end - start;
        
        // Save the baseline runtime (for 1 thread).
        if (num_threads == 1) {
            baseline_std = runtime_std;
            baseline_col = runtime_col;
        }
        
        // Compute efficiency: (speedup / num_threads) where speedup = baseline_runtime / runtime.
        double eff_std = (baseline_std / runtime_std) / num_threads;
        double eff_col = (baseline_col / runtime_col) / num_threads;
        
        results_std.push_back({"MatMul Standard", num_threads, runtime_std, eff_std});
        results_col.push_back({"MatMul Collapse(2)", num_threads, runtime_col, eff_col});
        
        delete[] A;
        delete[] B;
        delete[] C_std;
        delete[] C_col;
    }
    
    // Print results.
    cout << "\nMatMul Standard Results:\n";
    cout << "Version, Threads, Runtime, Efficiency\n";
    cout << fixed << setprecision(5);
    for (const auto &res : results_std) {
        cout << left
             << setw(20) << res.version 
             << setw(20) << res.num_threads 
             << setw(20) << res.runtime
             << setw(20) << res.efficiency 
             << endl;
    }
    
    cout << "\nMatMul Collapse(2) Results:\n";
    cout << "Version, Threads, Runtime, Efficiency\n";
    for (const auto &res : results_col) {
        cout << left
             << setw(20) << res.version 
             << setw(20) << res.num_threads 
             << setw(20) << res.runtime
             << setw(20) << res.efficiency 
             << endl;
    }
    std::cout << std::string(20 + 10 + 12 + 12 + 10, '-') << std::endl;
}

// Weak scaling experiment: the matrix size increases with the number of threads
// so that the work per thread remains roughly constant.
void run_weak_scaling_experiment() {
    vector<Result> results_std;
    vector<Result> results_col;
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    double baseline_std = 0.0, baseline_col = 0.0;
    
    cout << "Weak Scaling Experiment (Base Matrix Size = " << WEAK_BASE_N 
         << " x " << WEAK_BASE_N << " for 1 thread)\n";
    
    for (int t = 0; t < num_tests; t++) {
        int num_threads = thread_counts[t];
        omp_set_num_threads(num_threads);
        
        // Compute matrix dimension for weak scaling:
        int n = static_cast<int>(WEAK_BASE_N * static_cast<double>(num_threads));
        // Round up to a multiple of BLOCK_SIZE.
        if (n % BLOCK_SIZE != 0) {
            n = ((n / BLOCK_SIZE) + 1) * BLOCK_SIZE;
        }
        
        // Allocate matrices.
        double* A      = new double[n * n];
        double* B      = new double[n * n];
        double* C_std  = new double[n * n];
        double* C_col  = new double[n * n];
        
        // Initialize matrices.
        init_matrix(A, n, 1.0);
        init_matrix(B, n, 1.0);
        init_matrix(C_std, n, 0.0);
        init_matrix(C_col, n, 0.0);
        
        double start, end;
        // Run standard version.
        start = omp_get_wtime();
        cache_block_matmul_standard(A, B, C_std, n);
        end = omp_get_wtime();
        double runtime_std = end - start;
        
        // Run collapse(2) version.
        start = omp_get_wtime();
        cache_block_matmul_collapse(A, B, C_col, n);
        end = omp_get_wtime();
        double runtime_col = end - start;
        
        // Save the baseline runtime (for 1 thread).
        if (num_threads == 1) {
            baseline_std = runtime_std;
            baseline_col = runtime_col;
        }
        
        double eff_std = (baseline_std / runtime_std) / num_threads;
        double eff_col = (baseline_col / runtime_col) / num_threads;
        
        results_std.push_back({"MatMul Standard", num_threads, runtime_std, eff_std});
        results_col.push_back({"MatMul Collapse(2)", num_threads, runtime_col, eff_col});
        
        delete[] A;
        delete[] B;
        delete[] C_std;
        delete[] C_col;
    }
    
    // Print results for the standard version.
    cout << "\nMatMul Standard Results:\n";
    cout << "Version, Threads, Matrix Dimension, Runtime, Efficiency\n";
    cout << fixed << setprecision(5);
    for (int i = 0; i < num_tests; i++) {
        int num_threads = thread_counts[i];
        int n = static_cast<int>(WEAK_BASE_N * cbrt(static_cast<double>(num_threads)));
        if (n % BLOCK_SIZE != 0) {
            n = ((n / BLOCK_SIZE) + 1) * BLOCK_SIZE;
        }
        cout << left
             << setw(20) << results_std[i].version 
             << setw(20) << num_threads
             << setw(20) << results_std[i].runtime 
             << setw(20) << results_std[i].efficiency
             << endl;
        
    }
    
    // Print results for the collapse(2) version.
    cout << "\nMatMul Collapse(2) Results:\n";
    cout << "Version, Threads, Matrix Dimension, Runtime, Efficiency\n";
    for (int i = 0; i < num_tests; i++) {
        int num_threads = thread_counts[i];
        int n = static_cast<int>(WEAK_BASE_N * cbrt(static_cast<double>(num_threads)));
        if (n % BLOCK_SIZE != 0) {
            n = ((n / BLOCK_SIZE) + 1) * BLOCK_SIZE;
        }
        cout << left
             << setw(20) << results_col[i].version 
             << setw(20) << num_threads
             << setw(20) << results_col[i].runtime 
             << setw(20) << results_col[i].efficiency
             << endl;
    }
}