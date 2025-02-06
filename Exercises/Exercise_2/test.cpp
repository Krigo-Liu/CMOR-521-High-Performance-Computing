#include <iostream>
#include <chrono>
#include <cstring> // For memset

using namespace std;
using namespace std::chrono;

void matmul(const int n, double *C, const double *A, const double *B) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double Cij = 0.0; // Initialize Cij to 0
            for (int k = 0; k < n; ++k) {
                Cij += A[k + i * n] * B[j + k * n];
            }
            C[j + i * n] = Cij;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <matrix_size>" << endl;
        return 1;
    }

    int n = atoi(argv[1]);
    cout << "Matrix size n = " << n << endl;

    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C = new double[n * n];

    // Initialize A and B (example: identity matrices)
    memset(A, 0, n * n * sizeof(double));
    memset(B, 0, n * n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        A[i + i * n] = 1.0;
        B[i + i * n] = 1.0;
    }

    double min_time = 1e10;
    int num_samples = 25;
    for (int i = 0; i < num_samples; ++i) {
        memset(C, 0, n * n * sizeof(double)); // Reset C to zero before each matmul
        auto start = high_resolution_clock::now();
        matmul(n, C, A, B);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        min_time = min(static_cast<double>(duration.count()), min_time);
    }
    cout << "Elapsed time for matmul in (microsec): " << min_time << endl;

    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}