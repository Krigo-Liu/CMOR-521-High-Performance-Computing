#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

// computes b += A * x, assumes row major storage
void matvec(const int n, double *b, const double *A, const double *x)
{
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            b[i] += A[j + i * n] * x[j];
        }
    }
}

int main(int argc, char *argv[])
{

    int n = atoi(argv[1]);
    cout << "Matrix size n = " << n << endl;

    double *A = new double[n * n];
    double *x = new double[n];
    double *b = new double[n];

    double min_time = 1e10;
    int num_samples = 25;
    for (int i = 0; i < num_samples; ++i)
    {
        auto start = high_resolution_clock::now();
        matvec(n, b, A, x);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        min_time = std::min((double)duration.count(), min_time);
    }
    cout << "Elapsed time for matvec in (microsec): " << min_time << endl;

    delete[] A;
    delete[] b;
    delete[] x;

    return 0;
}
