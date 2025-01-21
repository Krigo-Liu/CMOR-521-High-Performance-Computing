#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace chrono;

double matsum_row_colunm(int n, double *A)
{
    double sum = 0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
          sum += A[j + i * n];
        }
    }
    return sum;
}

double matsum_colunm_row(int n, double *A)
{
    double sum = 0;
    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
            sum += A[j + i * n];
        }
    }
    return sum;  
}

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    cout << "n = " << n << endl;

    double *A = new double[n * n];
    for (int i = 0; i < n * n; ++i)
    {
        A[i] = 1.0 / (n * n);
    }

    int sum_trial = 25;
    double val_exact = 1.0;
    double tol = 1.0e-15 * n * n;

    double min_time_row_colunm = 1.0e9;
    for (int i = 0; i < sum_trial; ++i)
    {
        high_resolution_clock::time_point start = high_resolution_clock::now();
        double val = matsum_row_colunm(n, A);
        high_resolution_clock::time_point end = high_resolution_clock::now();  
        duration<double> time_span = (end - start);
        min_time_row_colunm = std::min((double) time_span.count(), min_time_row_colunm);
        if (fabs(val - val_exact) > tol)
        {
            cout << "matsum_row_colunm: wrong sum" << endl;
        }
    }

    // Measeure performance of matsum_colunm_row
    double min_time_colunm_row = 1.0e10;
    for (int i = 0; i < sum_trial; ++i)
    {
        high_resolution_clock::time_point start = high_resolution_clock::now();
        double val = matsum_colunm_row(n, A);
        high_resolution_clock::time_point end = high_resolution_clock::now();
        duration<double> time_span = (end - start);
        min_time_colunm_row = std::min((double) time_span.count(), min_time_colunm_row);

        if (fabs(val - val_exact) > tol)
        {
            cout << "matsum_colunm_row: wrong sum" << endl;
        }
    }

    cout << "row column elapsed time: " << min_time_row_colunm << " seconds" << endl;
    cout << "column row elapsed time: " << min_time_colunm_row << " seconds" << endl;
    
    delete[] A;
    return 0;  
}