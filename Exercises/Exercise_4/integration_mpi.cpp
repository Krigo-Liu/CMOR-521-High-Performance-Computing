#include <mpi.h>
#include <iostream>
#include <iomanip>

using namespace std;

int main(){
    double sum = 0.0;
    int num_steps = 100000000;
    double step = 1.0 / (double) num_steps;

    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double elapsed_time = MPI_Wtime();
    for (int i = rank; i < num_steps; i += size){
        double x = (i + 0.5) * step;
        sum = sum + 4.0 / (1.0 + x * x);
    }
    double pi = step * sum;
    MPI_Reduce(&pi, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    elapsed_time = MPI_Wtime() - elapsed_time;
    cout << "pi = " << setprecision(7) << sum << 
        " in " << elapsed_time << " secs" << " in rank " << rank << endl;
    MPI_Finalize();
    return 0;
}