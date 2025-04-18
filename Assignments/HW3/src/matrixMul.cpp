#include "../include/matrixMul.hpp"

// Check and generate matrics with the given parameters
void generateMatrices(const int M, const int N, const int P, const int k){

}

// Naive matrix multiplication: C = A * B (A: MxN, B: NxP, C: MxP)
void serialMatMult(const int N, double* C, const double* A, const double* B){
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            double sum = 0.0;
            for (int k = 0; k < N; k++){
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Test the serial result is equal to MPI result. (serialC: MxP, mpiC: MxP)
void testMul(const int N, double* serialC, double* mpiC){
    double error = 0.0;
    for (int i = 0; i < N * N; i++){
        error += abs(mpiC[i] - serialC[i]);
    }

    cout << "Total error = " << error << endl;
    if (error < 1e-6)
        cout << "SUMMA multiplication is CORRECT." << endl;
    else
        cout << "SUMMA multiplication is INCORRECT!" << endl;
}