#include "./include/matrixMul.hpp"

int main(int argc, char *argv[])
{
    // Process any command-line arguments relevant to MPI
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            cerr << "You need to use: mpirun -n <p> ./summa_mpi <N> <k> \n";
            cerr << "  p must be a perfect square, N divisible by k*sqrt(p)" << endl;
            MPI_Finalize();
            return -1;
        }
    }

    // Require that the number of processes is a perfect square
    int q = (int) sqrt(size);
    if (q * q != size) {
        if (rank == 0){
            cerr << "Error: Number of processes must be a perfect square." << endl;
            MPI_Finalize();
            return -1;
        }
    }

    int N = atoi(argv[1]);
    int k = atoi(argv[2]);
    int blockSize = N / q;
    if(blockSize % k != 0) {
        if(rank == 0){
            cerr << "Error: Block size must be divisible by k.\n";
            MPI_Finalize();
            return -1;
        }
    }
    
    // Create communicators for each row and column
    int row_color = rank / q;
    int col_color = rank % q;

    MPI_Comm row_comm, col_comm;
    MPI_Comm_split(MPI_COMM_WORLD, row_color, col_color, &row_comm);
    MPI_Comm_split(MPI_COMM_WORLD, col_color, row_color, &col_comm);

    // Allocate matrices A, B, C, and a reference matrix C_excepted;
    double *A = new double[N * N];
    double *B = new double[N * N];
    double *C = new double[N * N];
    double *C_expected = new double[N * N];

    double *A_block = new double[blockSize * blockSize];
    double *B_block = new double[blockSize * blockSize];
    double *C_block = new double[blockSize * blockSize];

    // Randomly initialize A and B
    if (rank == 0) {
        for (int i = 0; i < N*N; ++i){
            A[i] = rand() / double(RAND_MAX);
            B[i] = rand() / double(RAND_MAX);
        }
    }

    // blockSize = N / q
    int blockElems = blockSize * blockSize;
    double *A_packer = new double[blockElems * size];
    double *B_packer = new double[blockElems * size];

    if (rank == 0){
        for (int p = 0; p < size; ++p){
            int i = p / q, j = p % q;
            for (int r = 0; r < blockSize; ++r) {
                for (int c = 0; c < blockSize; ++c) {
                    A_packer[p * blockElems + r * blockSize + c] = 
                    A[(i * blockSize + r) * N + (j * blockSize + c)];
                    B_packer[p * blockElems + r * blockSize + c] = B[(i * blockSize + r) * N + (j * blockSize + c)];
                    
                }
            }
        }
    }
    MPI_Scatter(
        A_packer, blockElems, MPI_DOUBLE,
        A_block, blockElems, MPI_DOUBLE,
        0, MPI_COMM_WORLD    
    );
    MPI_Scatter(
        B_packer, blockElems, MPI_DOUBLE,
        B_block, blockElems, MPI_DOUBLE,
        0, MPI_COMM_WORLD    
    );

    //blockSize = N / q
    int microSteps = blockSize / k;

    // Main SUMMA outer product microstep loops
    // q = sqrt(size)
    double *A_panel = new double[blockSize * k];
    double *B_panel = new double[k * blockSize];
    for (int t = 0; t < q; ++t){
        for (int u = 0; u < microSteps; ++u) {
            // Pack A_panel on row-roots
            if (col_color == t) {
                for (int r = 0; r < blockSize; ++r){
                    for (int c = 0; c < k; ++c) {
                        A_panel[r * k + c] = A_block[r * blockSize + u * k + c];
                    }
                }
            }
            MPI_Bcast(
                A_panel, blockSize * k, MPI_DOUBLE, t, row_comm
            );

            // Pack B_panel on column-roots
            if (row_color == t) {
                for (int r = 0; r < k; ++r){
                    for (int c = 0; c < blockSize; ++c){
                        B_panel[r * blockSize + c] = B_block[(u * k + r) * blockSize + c];
                    }
                }
            }
            MPI_Bcast(
                B_panel, k * blockSize, MPI_DOUBLE, t, col_comm
            );

            // Local multiply-accumulate: C_block += A_panel * B_panel
            for (int i = 0; i < blockSize; ++i) {
                for (int j = 0; j < blockSize; ++j) {
                    double sum = 0.0;
                    for (int x = 0; x < k; ++x) {
                        sum += A_panel[i * k + x] * B_panel[x * blockSize + j];
                    }
                    C_block[i * blockSize + j] += sum;
                }
            }

        }
    }

    double *C_packer = new double[size * blockElems];
    MPI_Gather(
        C_block, blockElems, MPI_DOUBLE,
        C_packer, blockElems, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            int i = p / q, j = p % q;
            for (int r = 0; r < blockSize; ++r){
                for (int c = 0; c < blockSize; ++c){
                    C[(i * blockSize + r) * N + (j * blockSize + c)] = C_packer[p * blockElems + r * blockSize + c];
                }
            }
        }

        // Compute serial
        serialMatMult(N, C_expected, A, B);

        // Compare
        testMul(N, C_expected, C);

        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_expected;

        delete[] A_block;
        delete[] B_block;
        delete[] C_block;
    }

    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&col_comm);
    MPI_Finalize();
    return 0;
}