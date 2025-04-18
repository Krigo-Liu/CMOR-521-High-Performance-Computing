#include "mpi.h"
#include <iostream>

using namespace std;

void print_vector(int n, double *x)
{
  cout << "x = [";
  for (int i = 0; i < n - 1; ++i)
  {
    cout << x[i] << ", ";
  }
  cout << x[n - 1] << "]" << endl;
}

int main()
{

  MPI_Init(NULL, NULL);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Status status;

  int n = 10;
  double *x = new double[n];

  for (int i = 0; i < n; ++i)
  {
    x[i] = (double)rank + i;
  }

  if (rank == 0)
  {
    MPI_Send(x, n / 2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  }
  else if (rank == 1)
  {
    MPI_Recv(x + n / 2, n, MPI_DOUBLE, 0, MPI_ANY_TAG,
             MPI_COMM_WORLD, &status);
    int count;             
    MPI_Get_count(&status, MPI_DOUBLE, &count);
    cout << "on rank 1 with count " << count << endl; 
    // cout << "on rank " << rank << endl; 
    // print_vector(n, x);
  }

  MPI_Finalize();
  return 0;
}
