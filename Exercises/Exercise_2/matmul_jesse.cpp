#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

// computes C = C + A * B
void matmul_naive(const int n, double *C, double *A, double *B)
{
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < n; ++j)
    {
      double Cij = C[j + i * n];
      for (int k = 0; k < n; ++k)
      {
        // C_ij <-- A_ik * B_kj
        double Aij = A[k + i * n];
        double Bjk = B[j + k * n];
        //double Bjk = B[k + j * n]; // col major
        Cij += Aij * Bjk;
      }
      C[j + i * n] = Cij;
    }
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
      matmul_naive(n, C, A, B);
      auto end = high_resolution_clock::now();
      auto duration = duration_cast<microseconds>(end - start);
      min_time = std::min((double)duration.count(), min_time);
  }
  cout << "Elapsed time for matmul in (microsec): " << min_time << endl;

  delete[] A;
  delete[] B;
  delete[] C;

  return 0;
}
