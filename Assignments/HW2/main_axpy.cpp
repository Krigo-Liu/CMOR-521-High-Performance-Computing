#include "./include/axpy_omp.hpp"


struct Result{
    std::string version;
    int num_threads;
    double runtime;
    double efficiency;
};

int main(int argc, char *argv[]){
    vector<Result> result_version_1;
    vector<Result> result_version_2;
    // Parameters
    int n = 1000000000;
    int max_threads = 16;
    double alpha = 2.0;

    // Allocate arrays
    double *y = new double[n];
    double *x = new double[n];

    for(int threads = 1; threads <= max_threads; threads*=2){
        double runtime_version_1 = axpy_version_1_return(n, y, alpha, x, threads);
        double runtime_version_2 = axpy_version_2_return(n, y, alpha, x, threads);
        if (threads == 1){
            result_version_1.push_back({"AXPY Version 1", threads, runtime_version_1, 1.0});
            result_version_2.push_back({"AXPY Version 2", threads, runtime_version_2, 1.0});
        }
        else
        {
            double speedup_v1 = result_version_1[0].runtime / runtime_version_1;
            double speedup_v2 = result_version_2[0].runtime / runtime_version_2;
            double efficiency_v1 = speedup_v1 / threads;
            double efficiency_v2 = speedup_v2 / threads;
            result_version_1.push_back({"AXPY Version 1", threads, runtime_version_1, efficiency_v1});
            result_version_2.push_back({"AXPY Version 2", threads, runtime_version_2, efficiency_v2});
        }
    }

    cout << std::left
         << std::setw(20) << "AXPY Version"
         << std::setw(20) << "Threads"
         << std::setw(20) << "Runtime"
         << std::setw(20) << "Efficiency"
         << std::endl;

    // ofstream AXPY_version_1("./results/AXPY_version_1.csv", ios::app);
    // if (AXPY_version_1.is_open()){
    //     cout << "AXPY Version, Threads, Runtime, Efficiency" << endl;
    //     for (auto r : result_version_1){
    //         cout << r.version << ", " << r.num_threads << ", " << r.runtime << ", " << r.efficiency << endl;
    //     }
    //     AXPY_version_1.close();
    // }
    for (auto r : result_version_1){
        cout << std::left
             << std::setw(20) << r.version
             << std::setw(20) << r.num_threads
             << std::setw(20) << r.runtime
             << std::setw(20) << r.efficiency
             << std::endl;
    }

    std::cout << std::string(20 + 10 + 12 + 12 + 10, '-') << std::endl;

    // ofstream AXPY_version_2("./results/AXPY_version_2.csv", ios::app);
    // if (AXPY_version_2.is_open()){
    //     cout << "AXPY Version, Threads, Runtime, Efficiency" << endl;
    //     for (auto r : result_version_2){
    //         cout << r.version << ", " << r.num_threads << ", " << r.runtime << ", " << r.efficiency << endl;
    //     }
    //     AXPY_version_2.close();
    // }
    for (auto r : result_version_2){
        cout << std::left
             << std::setw(20) << r.version
             << std::setw(20) << r.num_threads
             << std::setw(20) << r.runtime
             << std::setw(20) << r.efficiency
             << std::endl;
    }
    
    // for(int i = 1; i < num_threads; i*=2;){
    //     axpy_version_2(n, y, alpha, x, i);
    // }

    delete[] x;
    delete[] y;
    return 0;
}

