#ifndef MATRIX_TRANSPOSE_HPP
#define MATRIX_TRANSPOSE_HPP

#include <vector>
#include <fstream>
#include <cmath>
#include <iostream>
#include <functional>

// Naive
void transpose_naive(const int n, double *AT, double *A);

// Cache-blocked
void transpose_blocked(const int n, double *AT, double *A, int block_size);

// Recursive
void transpose_recursive(const int n, double *AT, double *A, int threshold);

// Timing function using std::function
void time_transpose(const std::function<void(const int, double*, double*)> &transpose,
                    const int n, double *AT, double *A, double &time);

// Analyze naive transpose
void analyze_naive(int n, std::vector<double> &A, std::vector<double> &AT);

// Analyze different block sizes
void analyze_block_sizes(int n, std::vector<double> &A, std::vector<double> &AT);

// Analyze different recursion thresholds
void analyze_threshold_sizes(int n, std::vector<double> &A, std::vector<double> &AT);

#endif // MATRIX_TRANSPOSE_HPP
