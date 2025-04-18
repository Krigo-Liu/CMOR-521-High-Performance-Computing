#include "./include/matrixMul.hpp"

int main(int argc, char *argv[]){

    // Process any command-line arguments relevant to MPI
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) {
            cerr << "You need to use: mpirun -n <p> ./summa_mpi <N> \n";
            cerr << "  p must be a perfect square, N divisible by sqrt(p)" << endl;
            MPI_Finalize();
            return -1;
        }
    }

    // Require that the number of processes is a perfect square
    int q = (int) sqrt(size);
    int N = atoi(argv[1]);
    if(q*q != size || (N % q)!=0) {
        if (rank == 0){
            cerr << "Error: Number of processes must be a perfect square." << endl;
            MPI_Finalize();
            return -1;
        }
    }

    int blockSize = N / q;

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

    if(rank == 0) {
        delete[] A_packer;
        delete[] B_packer;
    }

    // Create a 2D toroidal cartesian communicator
    MPI_Comm cart;
    int dims[2]    = { q, q };   // number of processes in each dimension: rows = q, columns = q
    int periods[2] = { 1, 1 };   // periodicity flag: 1 = wrap-around (toroidal) in that dimension
    MPI_Cart_create(
        MPI_COMM_WORLD,           // input communicator
        2,                        // number of dimensions (2D grid)
        dims,                     // size of the grid in each dimension
        periods,                  // whether each dimension is periodic
        /*reorder=*/ 1,           // allow MPI to reorder ranks for efficiency
        &cart                     // output: new Cartesian communicator
    );
    
    // Get my coords and neighbors
    int coords[2];
    MPI_Cart_coords(cart, rank, 2, coords);
    int srcA, dstA, srcB, dstB;
    // dimension 1 = columns → shift for A (left/right)
    MPI_Cart_shift(cart, /*dim=*/1, /*disp=*/1, &srcA, &dstA);
    // dimension 0 = rows → shift for B (up/down)
    MPI_Cart_shift(cart, /*dim=*/0, /*disp=*/1, &srcB, &dstB);

    for (int i = 0; i < coords[0]; ++i){
        MPI_Sendrecv_replace(
            A_block, 
            blockElems, 
            MPI_DOUBLE,
            srcA,
            0,
            dstA,
            0,
            cart,
            MPI_STATUS_IGNORE
        );
    }

    for (int i = 0; i < coords[1]; ++i){
        MPI_Sendrecv_replace(
            B_block, 
            blockElems, 
            MPI_DOUBLE,
            srcB,
            0,
            dstB,
            0,
            cart,
            MPI_STATUS_IGNORE
        );
    }

    // Main Cannon loop: q steps
    for (int step = 0; step < q; ++step){
        // C_block += A_block * B_block
        for(int i = 0; i < blockSize; ++i){
            for(int j = 0; j < blockSize; ++j){
                for (int k = 0; k < blockSize; ++k){
                    C_block[i * blockSize + k] += A_block[i * blockSize + j] * B_block[j * blockSize + k];
                }
            }
        }

        // Shift A left by 1, B up by 1
        MPI_Sendrecv_replace(
            A_block,
            blockElems,
            MPI_DOUBLE,
            srcA,
            0,
            dstA,
            0,
            cart,
            MPI_STATUS_IGNORE
        );
        MPI_Sendrecv_replace(
            B_block,
            blockElems,
            MPI_DOUBLE,
            srcB,
            0,
            dstB,
            0,
            cart,
            MPI_STATUS_IGNORE
        );
    }

    // Gather C blocks back to root
    double *C_packer = new double[size * blockElems]; 
    MPI_Gather(
        C_block,
        blockElems,
        MPI_DOUBLE,
        C_packer,
        blockElems,
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
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

    MPI_Comm_free(&cart);
    MPI_Finalize();
    return 0;
}