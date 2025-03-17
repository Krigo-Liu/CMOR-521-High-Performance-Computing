#include "../include/axpy_omp.hpp"

void axpy_version_1(int n, double *y, double alpha, double *x) {
    double axpy_v1_time  = omp_get_wtime();
    int num_threads = omp_get_max_threads();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        for (int i = tid; i < n; i += num_threads) {
            y[i] += alpha * x[i];
        }
    }
    cout << "AXPY version 1 took " << omp_get_wtime() - axpy_v1_time << " seconds" << endl;
    cout << "AXPY version 1 took " << num_threads << " threads" << endl;
}

void axpy_version_1(int n, double *y, double alpha, double *x, int numberOfThreads) {
    double axpy_v1_time  = omp_get_wtime();
    int num_threads = omp_get_num_threads();
    #pragma omp parallel num_threads(numberOfThreads)
    {
        int tid = omp_get_thread_num();
        for (int i = tid; i < n; i += num_threads) {
            y[i] += alpha * x[i];
        }
    }
    cout << "AXPY version 1 took " << omp_get_wtime() - axpy_v1_time << " seconds" << endl;
    cout << "AXPY version 1 took " << numberOfThreads << " threads" << endl;
}

double axpy_version_1_return(int n, double *y, double alpha, double *x) {
    double axpy_v1_time  = omp_get_wtime();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        for (int i = tid; i < n; i += num_threads) {
            y[i] += alpha * x[i];
        }
    }
    return omp_get_wtime() - axpy_v1_time;
}

double axpy_version_1_return(int n, double *y, double alpha, double *x, int numberOfThreads) {
    double axpy_v1_time  = omp_get_wtime();
    #pragma omp parallel num_threads(numberOfThreads)
    {
        int tid = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        for (int i = tid; i < n; i += num_threads) {
            y[i] += alpha * x[i];
        }
    }
    return omp_get_wtime() - axpy_v1_time;
}

//-------- Version 2 --------

void axpy_version_2(int n, double *y, double alpha, double *x) {
    double axpy_v2_time = omp_get_wtime();
    int num_threads = omp_get_num_threads();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int chunk_size = n / num_threads;
        int start = tid * chunk_size;
        int end = (tid == num_threads - 1) ? n : start + chunk_size;

        for (int i = start; i < end; i++) {
            y[i] += alpha * x[i];
        }
    }
    cout << "AXPY version 2 took " << omp_get_wtime() - axpy_v2_time << " seconds" << endl;

    cout << "AXPY version 2 took " << num_threads << " threads" << endl;
}

void axpy_version_2(int n, double *y, double alpha, double *x, int numberOfThreads) {
    double axpy_v2_time = omp_get_wtime();
    #pragma omp parallel num_threads(numberOfThreads)
    {
        int tid = omp_get_thread_num();
        int chunk_size = n / numberOfThreads;
        int start = tid * chunk_size;
        int end = (tid == numberOfThreads - 1) ? n : start + chunk_size;

        for (int i = start; i < end; i++) {
            y[i] += alpha * x[i];
        }
    }
    cout << "AXPY version 2 took " << omp_get_wtime() - axpy_v2_time << " seconds" << endl;

    cout << "AXPY version 2 took " << numberOfThreads << " threads" << endl;
}

double axpy_version_2_return(int n, double *y, double alpha, double *x) {
    double axpy_v2_time = omp_get_wtime();
    int num_threads = omp_get_num_threads();
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int chunk_size = n / num_threads;
        int start = tid * chunk_size;
        int end = (tid == num_threads - 1) ? n : start + chunk_size;

        for (int i = start; i < end; i++) {
            y[i] += alpha * x[i];
        }
    }
    return omp_get_wtime() - axpy_v2_time;
}

double axpy_version_2_return(int n, double *y, double alpha, double *x, int numberOfThreads) {
    double axpy_v2_time = omp_get_wtime();
    #pragma omp parallel num_threads(numberOfThreads)
    {
        int tid = omp_get_thread_num();
        int chunk_size = n / numberOfThreads;
        int start = tid * chunk_size;
        int end = (tid == numberOfThreads - 1) ? n : start + chunk_size;

        for (int i = start; i < end; i++) {
            y[i] += alpha * x[i];
        }
    }
    return omp_get_wtime() - axpy_v2_time;
}