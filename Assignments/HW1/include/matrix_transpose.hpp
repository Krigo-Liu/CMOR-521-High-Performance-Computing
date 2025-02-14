#ifndef MATRIX_MULTIPLICATION_HPP
#define MATRIX_MULTIPLICATION_HPP

#include <iostream>
#include <chrono>
#include <cmath>
#include <functional>
#include <fstream>
#include <vector>

using namespace std;

// Default BLOCK_SIZE, can be changed
extern int BLOCK_SIZE;

// Default threshold, can be changed
extern int threshold;

// Check if the header has been written
extern bool header_written;

// Naive
void transpose_naive(const int n, double *AT, double *A);

// Cache-blocked
void transpose_blocked(const int n, double *AT, double *A, int block_size);

// Recursive
void transpose_recursive(const int n, double *AT, double *A, int threshold);

// Timing function using std::function
void time_transpose(
    const std::function<void(const int, double*, double*)> &transpose,
    const int n,
    double *AT,
    const double *A,
    double &time
);

// Analyze naive transpose
double analyze_naive(const int n, double * AT, double * A);

// Analyze different block sizes
double analyze_block(const int n, double * AT, double * A);

// Analyze different recursion thresholds
double analyze_recursive(const int n, double * AT, double * A);

// Change the block size
void change_block_size(int block_size);

// Change the threshold
void change_threshold(int threshold_size);


// Avoid writing the header multiple times
void already_header_written();

// Function to compute relative error
double compute_relative_error(const double * AT_expected, const double * AT_computed, int n);

// Function to check if the relative error is within machine precision
void check_transpose_accuracy(const double * AT_expected, const double * AT_computed, int n);


#endif // MATRIX_MULTIPLICATION_HPP

