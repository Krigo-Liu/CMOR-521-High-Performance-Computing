#include "mpi.h"
#include <iostream>

using namespace std;

int main()
{

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // compute y = a * b' * x
    int n_local = 4;
    double *x = new double[n_local];
    double *a = new double[n_local];
    double *b = new double[n_local];
    for (int i = 0; i < n_local; i++){
        a[i] = 1.0;
        b[i] = 1.0;
        x[i] = i + rank * n_local; 
    }

    cout << "on rank " << rank << ", x = \n";
    for (int i = 0; i < n_local; ++i)
        cout << x[i] << "\n";
    cout << endl; 

    // local sum
    double b_dot_x_local = 0.0; 
    for (int i = 0; i < n_local; ++i)
        b_dot_x_local += b[i] * x[i];

    double b_dot_x;        
    MPI_Allreduce(&b_dot_x_local, &b_dot_x, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    cout << "on rank " << rank << ", y = \n";
    for (int i = 0; i < n_local; ++i)
        cout << a[i] * b_dot_x << "\n";
    cout << endl; 

    MPI_Finalize();

    return 0;
}
