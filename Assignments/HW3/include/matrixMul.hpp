#ifndef MATRIX_MUL_HPP
#define MATRIX_MUL_HPP

#include <iostream>
#include <mpi.h>
#include <random>
#include <cmath>
#include <cassert>
#include <cstdlib>


using namespace std;

// Test the matrix multiplication equal to naive version
void testMul(const int N, double* serialC, double* mpiC);

// Naive matrix multiplication
void serialMatMult(const int N, double* C, const double* A, const double* B);

#endif