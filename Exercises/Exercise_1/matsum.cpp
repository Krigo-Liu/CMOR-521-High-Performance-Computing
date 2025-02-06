#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

double matsum_ij(int n, double *A)
{
  double val = 0.0;
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      val += A[j + i * n];
    }
  }
  return val;
}

double matsum_ji(int n, double *A)
{
  double val = 0.0;
  for (int j = 0; j < n; ++j)
  {
    for (int i = 0; i < n; ++i)
    {
      val += A[j + i * n];
    }
  }
  return val;
}

int main(int argc, char *argv[])
{

  int n = atoi(argv[1]);
  cout << "Matrix size n = " << n << endl;

  double *A = new double[n * n];

  for (int i = 0; i < n * n; ++i)
  {
    A[i] = 1.0 / (n * n);
  }

  int num_trials = 25;
  double val_exact = 1.0;
  double tol = 1e-15 * n * n;

  // Measure performance
  double min_time_ij = 1e9;
  for (int i = 0; i < num_trials; ++i)
  {
    high_resolution_clock::time_point start = high_resolution_clock::now();
    double val = matsum_ij(n, A);
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> duration = (end - start);
    min_time_ij = std::min((double) duration.count(), min_time_ij);

    if (fabs(val - val_exact) > tol)
    {
      cout << "you did something wrong " << endl;
    }
  }
  

  // Measure performance
  double min_time_ji = 1e9;
  for (int i = 0; i < num_trials; ++i)
  {
    high_resolution_clock::time_point start = high_resolution_clock::now();
    double val = matsum_ji(n, A);
    high_resolution_clock::time_point end = high_resolution_clock::now();
    duration<double> duration = (end - start);
    min_time_ji = std::min((double) duration.count(), min_time_ji);

    if (fabs(val - val_exact) > tol)
    {
      cout << "you did something wrong " << endl;
    }
  }

  cout << "ij elapsed time = " << min_time_ij << endl;
  cout << "ji elapsed time = " << min_time_ji << endl;

  delete[] A;

  return 0;
}
