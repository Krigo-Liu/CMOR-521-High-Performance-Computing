#include "mpi.h"
#include <iostream>

using namespace std;

int main()
{

    MPI_Init(NULL, NULL);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *x_send = new int[2 * size];
    for (int i = 0; i < 2 * size; i+=2){
        x_send[i] = rank + 2 * i * size;
        x_send[i+1] = rank + (2 * i + 1) * size;
    }

    cout << "on rank " << rank << ", x_send = ";
    for (int i = 0; i < 2 * size; ++i)
        cout << x_send[i] << ", ";
    cout << endl; 

    int *x_recv = new int[2 * size];
    MPI_Alltoall(x_send, 2, MPI_INT, x_recv, 2, MPI_INT, MPI_COMM_WORLD);

    if (rank == 0)
        cout << endl; 
    MPI_Barrier(MPI_COMM_WORLD);

    cout << "on rank " << rank << ", x_recv = ";
    for (int i = 0; i < 2 * size; ++i)
        cout << x_recv[i] << ", ";
    cout << endl;

    MPI_Finalize();

    return 0;
}
