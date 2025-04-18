#include "mpi.h"
#include <iostream>

using namespace std;

int main()
{

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // compute y = [A_1 | A_2 | A_3 ...] * x
    int n_local = 4;
    int n = n_local * size;
    double *x = new double[n_local];
    double *y = new double[n_local];
    double *A_local = new double[n * n_local]; // assume column major
    for (int i = 0; i < n_local; i++){
        for (int row = 0; row < n; ++row) // loop over rows
            A_local[row + i * n] = 1.0;

        x[i] = i + rank * n_local; 
    }

    cout << "on rank " << rank << ", x = \n";
    for (int i = 0; i < n_local; ++i)
        cout << x[i] << "\n";
    cout << endl; 

    // local product
    double *Ax_local = new double[n];
    for (int j = 0; j < n_local; j++)
        for (int row = 0; row < n; ++row) // loop over rows
            Ax_local[row] += A_local[row + j * n] * x[j];

    // combine local results            
    int * recvcount = new int[size];
    for (int i = 0; i < size; ++i)
        recvcount[i] = n_local;
    MPI_Reduce_scatter(Ax_local, y, recvcount, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    cout << "on rank " << rank << ", y = \n";
    for (int i = 0; i < n_local; ++i)
        cout << y[i] << "\n";
    cout << endl; 

    MPI_Finalize();

    return 0;
}
