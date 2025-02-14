#ifndef MATRIX_TRANSPOSE_HPP
#define MATRIX_TRANSPOSE_HPP

#include <iostream>
#include <chrono>
#include <cmath>
#include <functional>
#include <fstream>
#include <vector>
#include <cstdlib>       // For rand()
#include <limits>        // For numeric_limits

using namespace std;

// Default BLOCK_SIZE, can be changed
extern int BLOCK_SIZE;

// Default threshold, can be changed
extern int threshold;

// Check if the header has been written
extern bool header_written;

// Naive
void multiply_naive(const int n, double* C, const double* A, const double* B);

// Cache-blocked
void multiply_block(const int n, double* C, const double* A, const double* B);

// Recursive
void multiply_recursive(const int n, double* C, const double* A, const double* B);

// Timing function using std::function
void time_multiplication(
    const std::function<void(const int, double*, const double*, const double*)> &matmul,
    const int n,
    double *C,
    const double *A,
    const double *B,
    double &time
);

// Analyze naive transpose
double analyze_naive(const int n, double *C, const double *A, const double *B);

// Analyze different block sizes
double analyze_block(const int n, double *C, const double *A, const double *B, const double *C_expected);

// Analyze different recursion thresholds
double analyze_recursive(const int n, double *C, const double *A, const double *B, const double *C_expected);

// Change the block size
void change_block_size(int block_size);

// Change the threshold
void change_threshold(int threshold_size);

// Avoid writing the header multiple times
void already_header_written();

// Function to compute relative error
double compute_relative_error(const double* C_expected, const double* C_computed, int n);

// Function to check if the relative error is within machine precision
void check_multiplication_accuracy(const double* C_expected, const double* C_computed, int n);


#endif // MATRIX_TRANSPOSE_HPP
