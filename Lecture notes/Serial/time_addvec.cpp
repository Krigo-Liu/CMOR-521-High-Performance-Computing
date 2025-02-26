#include <cblas.h>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

// to compile, run "module load OpenBLAS" and then
//   g++ time_addvec.cpp -I${EBROOTOPENBLAS}/include/ -L${EBROOTOPENBLAS}/lib -lopenblas

// computes y = y + alpha * x
void add_vec(const int n, double *y, const double alpha, double *x)
{
    for (int i = 0; i < n; ++i)
    {
      y[i] = y[i] + alpha * x[i];
    }
}

int main(int argc, char *argv[])
{

    int n = atoi(argv[1]);
    int num_samples = 100;
    cout << "size n = " << n << 
      ", num timing samples = " << num_samples << endl;
    
    double *x = new double[n];
    double *y = new double[n];
    for (int i = 0; i < n; ++i){
      x[i] = 1.0;
      y[i] = 0.0;
    }
    double alpha = 1.0;
   
    double min_time = 1e10;
    for (int i = 0; i < num_samples; ++i)
    {
        auto start = high_resolution_clock::now();
        add_vec(n, y, alpha, x);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start);
        min_time = std::min((double)duration.count(), min_time);
    }
    cout << "Elapsed time for addvec in (nanosec): " << min_time << endl;
    
    double min_time_cblas = 1e10;
    for (int i = 0; i < num_samples; ++i)
    {
        auto start = high_resolution_clock::now();
	cblas_daxpy(n, alpha, x, 1, y, 1);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<nanoseconds>(end - start);
        min_time_cblas = std::min((double)duration.count(), min_time_cblas);
    }
    cout << "Elapsed time for cblas in (nanosec): " << min_time_cblas << endl;
        
    cout << y[0] << endl;
    delete[] x;
    delete[] y;

    return 0;
}
