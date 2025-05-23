#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

// computes C = C + A*B
void matmul_naive(const int n, double *C, double *A, double *B)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double Cij = C[j + i * n];
            for (int k = 0; k < n; ++k)
            {
                double Aij = A[k + i * n];
                double Bjk = B[j + k * n];
                Cij += Aij * Bjk;
            }
            C[j + i * n] = Cij;
        }
    }
}

#define BLOCK_SIZE 2 // bxb blocks

void matmul_blocked(const int n, double *C, double *A, double *B)
{
    for (int i = 0; i < n; i += BLOCK_SIZE)
    {
        for (int j = 0; j < n; j += BLOCK_SIZE)
        {
            for (int k = 0; k < n; k += BLOCK_SIZE)
            {

                // small matmul
                for (int ii = i; ii < i + BLOCK_SIZE; ii++)
                {
                    for (int jj = j; jj < j + BLOCK_SIZE; jj++)
                    {
                        double Cij = C[jj + ii * n];
                        for (int kk = k; kk < k + BLOCK_SIZE; kk++)
                        {
                            Cij += A[kk + ii * n] * B[jj + kk * n]; // Aik * Bkj
                        }
                        C[jj + ii * n] = Cij;
                    }
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{

    int n = atoi(argv[1]);
    cout << "Matrix size n = " << n << 
        ", block size = " << BLOCK_SIZE << endl;

    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C = new double[n * n];

    double min_time_naive = 1e10;
    int num_samples = 25;
    for (int i = 0; i < num_samples; ++i)
    {
        auto start = high_resolution_clock::now();
        matmul_naive(n, C, A, B);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        min_time_naive = std::min((double)duration.count(), min_time_naive);
    }
    cout << "Elapsed time for naive matmul in (microsec): " << min_time_naive << endl;

    double min_time_blocked = 1e10;
    for (int i = 0; i < num_samples; ++i)
    {
        auto start = high_resolution_clock::now();
        matmul_blocked(n, C, A, B);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        min_time_blocked = std::min((double)duration.count(), min_time_blocked);
    }
    cout << "Elapsed time for blocked matmul in (microsec): " << min_time_blocked << endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
