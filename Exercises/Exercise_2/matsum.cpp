#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

// compute C = A + B, assumes row major storage
void matsum(const int n, double *C, const double *A, const double *B)
{
    for (int i = 0; i < n * n; ++i)
    {
        double Aij = A[i];
        double Bij = B[i];
        C[i] = Aij + Bij;
    }
}

int main(int argc, char *argv[])
{

    int n = atoi(argv[1]);
    cout << "Matrix size n = " << n << endl;

    double *A = new double[n * n];
    double *B = new double[n * n];
    double *C = new double[n * n];

    double min_time = 1e10;
    int num_samples = 25;
    for (int i = 0; i < num_samples; ++i)
    {
        auto start = high_resolution_clock::now();
        matsum(n, C, A, B);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        min_time = std::min((double)duration.count(), min_time);
    }
    cout << "Elapsed time for matsum in (microsec): " << min_time << endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
