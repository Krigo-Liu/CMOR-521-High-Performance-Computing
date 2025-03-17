#include "./include/cache_block_matmul.hpp"


int STRONG_N    = 2048;  // Fixed matrix dimension for strong scaling
int WEAK_BASE_N = 128;   // Base matrix dimension for 1 thread in weak scaling
int BLOCK_SIZE  = 32;    // Block size for cache blocking

int main(int argc, char* argv[]) {
    
    vector<Result> results_std;
    vector<Result> results_col_2;
    vector<Result> results_col_3;
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    double baseline_std = 0.0, baseline_col_2 = 0.0, baseline_col_3 = 0.0;
    
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
        double runtime_col_2 = end - start;

        // Run the collapse(3) version.
        start = omp_get_wtime();
        cache_block_matmul_collapse_3(A, B, C_col, n);
        end = omp_get_wtime();
        double runtime_col_3 = end - start;

        // Save the baseline runtime (for 1 thread).
        if (num_threads == 1) {
            baseline_std = runtime_std;
            baseline_col_2 = runtime_col_2;
            baseline_col_3 = runtime_col_3;
        }
        
        // Compute efficiency: (speedup / num_threads) where speedup = baseline_runtime / runtime.
        double eff_std = (baseline_std / runtime_std) / num_threads;
        double eff_col_2 = (baseline_col_2 / runtime_col_2) / num_threads;
        double eff_col_3 = (baseline_col_3 / runtime_col_3) / num_threads;
        
        results_std.push_back({"MatMul Standard", num_threads, n, runtime_std, eff_std});
        results_col_2.push_back({"MatMul Collapse(2)", num_threads, n, runtime_col_2, eff_col_2});
        results_col_3.push_back({"MatMul Collapse(3)", num_threads, n, runtime_col_3, eff_col_3});
        
        delete[] A;
        delete[] B;
        delete[] C_std;
        delete[] C_col;
    }

    // Print results.
    cout << "\nMatMul Standard Results:\n";
    cout << std::left
         << std::setw(20) << "Version"
         << std::setw(20) << "Threads"
         << std::setw(20) << "Matrix"
         << std::setw(20) << "Runtime"
         << std::setw(20) << "Efficiency"
         << std::endl;    
    cout << fixed << setprecision(5);
    for (const auto &res : results_std) {
        cout << left
             << setw(20) << res.version 
             << setw(20) << res.numThreads
             << setw(20) << res.matrixSize
             << setw(20) << res.runtime
             << setw(20) << res.efficiency 
             << endl;
    }
    
    cout << "\nMatMul Collapse(2) Results:\n";
    cout << std::left
         << std::setw(20) << "Version"
         << std::setw(20) << "Threads"
         << std::setw(20) << "Matrix Size"
         << std::setw(20) << "Runtime"
         << std::setw(20) << "Efficiency"
         << std::endl; 
    for (const auto &res : results_col_2) {
        cout << left
             << setw(20) << res.version 
             << setw(20) << res.numThreads
             << setw(20) << res.matrixSize
             << setw(20) << res.runtime
             << setw(20) << res.efficiency 
             << endl;
    }

    cout << "\nMatMul Collapse(3) Results:\n";
    cout << std::left
         << std::setw(20) << "Version"
         << std::setw(20) << "Threads"
         << std::setw(20) << "Matrix Size"
         << std::setw(20) << "Runtime"
         << std::setw(20) << "Efficiency"
         << std::endl; 
    for (const auto &res : results_col_3) {
        cout << left
             << setw(20) << res.version 
             << setw(20) << res.numThreads
             << setw(20) << res.matrixSize
             << setw(20) << res.runtime
             << setw(20) << res.efficiency 
             << endl;
    }

    return 0;
}