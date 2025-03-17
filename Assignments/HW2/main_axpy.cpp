#include "./include/axpy_omp.hpp"


struct Result {
    string version;
    int num_threads;
    double runtime;
    double efficiency;
};

int main(int argc, char *argv[]){
    // Parameters for both experiments
    int max_threads = 16;
    double alpha = 2.0;

    /**********************
     * STRONG SCALING
     **********************/
    // In strong scaling, we fix the total problem size.
    int n_strong = 1000000000; // fixed problem size
    vector<Result> strong_result_v1;
    vector<Result> strong_result_v2;
    
    // Allocate arrays for strong scaling
    double *x_strong = new double[n_strong];
    double *y_strong = new double[n_strong];
    // (Optional) Initialize x_strong and y_strong here if needed.

    cout << "=== Strong Scaling ===" << endl;
    for(int threads = 1; threads <= max_threads; threads *= 2){
        double runtime_v1 = axpy_version_1_return(n_strong, y_strong, alpha, x_strong, threads);
        double runtime_v2 = axpy_version_2_return(n_strong, y_strong, alpha, x_strong, threads);
        if (threads == 1){
            strong_result_v1.push_back({"AXPY Version 1", threads, runtime_v1, 1.0});
            strong_result_v2.push_back({"AXPY Version 2", threads, runtime_v2, 1.0});
        }
        else {
            double speedup_v1 = strong_result_v1[0].runtime / runtime_v1;
            double speedup_v2 = strong_result_v2[0].runtime / runtime_v2;
            double efficiency_v1 = speedup_v1 / threads;
            double efficiency_v2 = speedup_v2 / threads;
            strong_result_v1.push_back({"AXPY Version 1", threads, runtime_v1, efficiency_v1});
            strong_result_v2.push_back({"AXPY Version 2", threads, runtime_v2, efficiency_v2});
        }
    }

    // Print Strong Scaling results
    cout << std::left << setw(20) << "AXPY Version"
         << setw(20) << "Threads"
         << setw(20) << "Runtime"
         << setw(20) << "Efficiency" << endl;
    for (auto r : strong_result_v1){
        cout << std::left << setw(20) << r.version
             << setw(20) << r.num_threads
             << setw(20) << r.runtime
             << setw(20) << r.efficiency << endl;
    }
    std::cout << std::string(20 + 10 + 12 + 12 + 10, '-') << std::endl;
    for (auto r : strong_result_v2){
        cout << std::left << setw(20) << r.version
             << setw(20) << r.num_threads
             << setw(20) << r.runtime
             << setw(20) << r.efficiency << endl;
    }

    // Free strong scaling arrays
    delete[] x_strong;
    delete[] y_strong;

    /**********************
     * WEAK SCALING
     **********************/
    // For weak scaling, we keep the workload per thread constant.
    // That is, we define a base workload per thread and set total n = base_n * num_threads.
    int base_n = 1000000000 / 16;  // workload per thread
    vector<Result> weak_result_v1;
    vector<Result> weak_result_v2;
    
    cout << "\n=== Weak Scaling ===" << endl;
    for(int threads = 1; threads <= max_threads; threads *= 2){
        int n_weak = base_n * threads; // total problem size increases with threads

        // Allocate new arrays for each weak scaling run.
        double *x_weak = new double[n_weak];
        double *y_weak = new double[n_weak];
        // (Optional) Initialize x_weak and y_weak here if needed.

        double runtime_v1 = axpy_version_1_return(n_weak, y_weak, alpha, x_weak, threads);
        double runtime_v2 = axpy_version_2_return(n_weak, y_weak, alpha, x_weak, threads);
        if (threads == 1){
            weak_result_v1.push_back({"AXPY Version 1", threads, runtime_v1, 1.0});
            weak_result_v2.push_back({"AXPY Version 2", threads, runtime_v2, 1.0});
        }
        else {
            // For weak scaling, the ideal is to keep runtime constant.
            // Thus, we compare against the single-thread runtime.
            double efficiency_v1 = weak_result_v1[0].runtime / runtime_v1;
            double efficiency_v2 = weak_result_v2[0].runtime / runtime_v2;
            weak_result_v1.push_back({"AXPY Version 1", threads, runtime_v1, efficiency_v1});
            weak_result_v2.push_back({"AXPY Version 2", threads, runtime_v2, efficiency_v2});
        }
        
        delete[] x_weak;
        delete[] y_weak;
    }

    // Print Weak Scaling results
    cout << std::left << setw(20) << "AXPY Version"
         << setw(20) << "Threads"
         << setw(20) << "Runtime"
         << setw(20) << "Efficiency" << endl;
    for (auto r : weak_result_v1){
        cout << std::left << setw(20) << r.version
             << setw(20) << r.num_threads
             << setw(20) << r.runtime
             << setw(20) << r.efficiency << endl;
    }
    std::cout << std::string(20 + 10 + 12 + 12 + 10, '-') << std::endl;
    for (auto r : weak_result_v2){
        cout << std::left << setw(20) << r.version
             << setw(20) << r.num_threads
             << setw(20) << r.runtime
             << setw(20) << r.efficiency << endl;
    }

    return 0;
}