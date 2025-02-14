#include "../include/matrix_multiplication.hpp"

using namespace std;

// ------------------------------------------------
// Change the BLOCK_SIZE
// ------------------------------------------------
void change_block_size(int block_size) {
    BLOCK_SIZE = block_size;
}

// ------------------------------------------------
// Change the threshold
// ------------------------------------------------
void change_threshold(int threshold_size) {
    threshold = threshold_size;
}

// ------------------------------------------------
// Mark the CSV header as written
// ------------------------------------------------
void already_header_written() {
    header_written = true;
}

// ------------------------------------------------
// Compute relative error between two matrices
// Used to compare the results of matrix multiplication
// ------------------------------------------------
double compute_relative_error(const double* C_expected, const double* C_computed, int n) {
    double max_relative_error = 0.0;
    double epsilon = numeric_limits<double>::epsilon();  // ~1e-16

    for (int i = 0; i < n * n; i++) {
        double abs_error = fabs(C_expected[i] - C_computed[i]);
        double denom     = fabs(C_expected[i]) + epsilon;  // Avoid division by zero
        double rel_error = abs_error / denom;

        max_relative_error = max(max_relative_error, rel_error);
    }
    return max_relative_error;
}

// ------------------------------------------------
// Check if multiplication results are acceptable
// ------------------------------------------------
void check_multiplication_accuracy(const double* C_expected, const double* C_computed, int n) {
    double relative_error = compute_relative_error(C_expected, C_computed, n);
    cout << "Max Relative Error: " << relative_error << endl;

    // Multiplications may accumulate more floating-point errors, so the tolerance can be relaxed.
    if (relative_error < numeric_limits<double>::epsilon() * 1e3) {
        cout << "✅ Multiplication implementation is sufficiently accurate.\n";
    } else {
        cout << "❌ WARNING: Multiplication implementation has significant numerical errors!\n";
    }
}

// ------------------------------------------------
// Naive matrix multiplication: C = A * B
// Dimensions: A, B, C are all (n x n), row-major order
// ------------------------------------------------
void multiply_naive(const int n, double* C, const double* A, const double* B) {
    // Initialize C to zero to avoid any leftover data
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0;
    }
    // Classic triple nested loops, O(n^3)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            double sum = 0.0;
            for (int k = 0; k < n; k++){
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// ------------------------------------------------
// Cache-blocked matrix multiplication
// Typical "block-based" multiplication
// ------------------------------------------------
void multiply_block(const int n, double* C, const double* A, const double* B) {
    // Initialize C to zero
    for (int i = 0; i < n * n; i++) {
        C[i] = 0.0;
    }

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                // Process elements within the smaller block
                for (int ii = i; ii < i + BLOCK_SIZE && ii < n; ii++) {
                    for (int jj = j; jj < j + BLOCK_SIZE && jj < n; jj++) {
                        double sum = C[ii * n + jj];
                        for (int kk = k; kk < k + BLOCK_SIZE && kk < n; kk++) {
                            sum += A[ii * n + kk] * B[kk * n + jj];
                        }
                        C[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
}

// ------------------------------------------------
// Recursive matrix multiplication
// Here we use a divide-and-conquer approach:
// - If the submatrix size <= threshold, use naive
// - Otherwise, split the matrices into four quadrants
//   and recursively compute
// ------------------------------------------------
static void multiply_recursive_helper(const int n,
                                      double* C, const double* A, const double* B,
                                      int rowA, int colA,
                                      int rowB, int colB,
                                      int rowC, int colC,
                                      int size) 
{
    if (size <= threshold) {
        // Directly do naive multiplication for small submatrix
        for (int i = 0; i < size; i++){
            for (int j = 0; j < size; j++){
                double sum = 0.0;
                for (int k = 0; k < size; k++){
                    double a_val = A[(rowA + i) * n + (colA + k)];
                    double b_val = B[(rowB + k) * n + (colB + j)];
                    sum += a_val * b_val;
                }
                C[(rowC + i) * n + (colC + j)] += sum;
            }
        }
    } else {
        // Split the submatrix into four blocks
        int half = size / 2;

        // C11 = A11*B11 + A12*B21
        multiply_recursive_helper(n, C, A, B,
                                 rowA, colA,         // A11 start
                                 rowB, colB,         // B11 start
                                 rowC, colC,         // C11 start
                                 half);
        multiply_recursive_helper(n, C, A, B,
                                 rowA, colA + half,   // A12 start
                                 rowB + half, colB,   // B21 start
                                 rowC, colC,         // C11 accumulate
                                 half);

        // C12 = A11*B12 + A12*B22
        multiply_recursive_helper(n, C, A, B,
                                 rowA, colA,         // A11 start
                                 rowB, colB + half,  // B12 start
                                 rowC, colC + half,  // C12 start
                                 half);
        multiply_recursive_helper(n, C, A, B,
                                 rowA, colA + half,   // A12 start
                                 rowB + half, colB + half,  // B22 start
                                 rowC, colC + half,  // C12 accumulate
                                 half);

        // C21 = A21*B11 + A22*B21
        multiply_recursive_helper(n, C, A, B,
                                 rowA + half, colA,    // A21 start
                                 rowB, colB,           // B11 start
                                 rowC + half, colC,    // C21 start
                                 half);
        multiply_recursive_helper(n, C, A, B,
                                 rowA + half, colA + half, // A22 start
                                 rowB + half, colB,        // B21 start
                                 rowC + half, colC,       // C21 accumulate
                                 half);

        // C22 = A21*B12 + A22*B22
        multiply_recursive_helper(n, C, A, B,
                                 rowA + half, colA,           // A21 start
                                 rowB, colB + half,           // B12 start
                                 rowC + half, colC + half,    // C22 start
                                 half);
        multiply_recursive_helper(n, C, A, B,
                                 rowA + half, colA + half,    // A22 start
                                 rowB + half, colB + half,    // B22 start
                                 rowC + half, colC + half,    // C22 accumulate
                                 half);
    }
}

void multiply_recursive(const int n, double* C, const double* A, const double* B) {
    // Initialize C to zero before recursion
    for (int i = 0; i < n * n; i++){
        C[i] = 0.0;
    }
    multiply_recursive_helper(n, C, A, B, 
                             0, 0,   // rowA, colA
                             0, 0,   // rowB, colB
                             0, 0,   // rowC, colC
                             n);
}

// ------------------------------------------------
// Timing function (uses std::function) to measure 
// the runtime of a given multiplication routine
// ------------------------------------------------
void time_multiplication(const std::function<void(const int, double*, const double*, const double*)> &matmul,
                         const int n,
                         double *C, const double *A, const double *B,
                         double &time)
{
    int trials = 25;
    double min_time = numeric_limits<double>::infinity();

    for (int t = 0; t < trials; t++) {
        clock_t start = clock();
        matmul(n, C, A, B);
        clock_t end = clock();

        double elapsed = double(end - start) / CLOCKS_PER_SEC;
        if (elapsed < min_time) {
            min_time = elapsed;
        }
    }
    time = min_time;
}

// ------------------------------------------------
// Analysis functions for naive, block, and recursive
// Return the measured time and compare with the reference result
// ------------------------------------------------
double analyze_naive(const int n, double *C, const double *A, const double *B) {
    double measure_time;
    time_multiplication(multiply_naive, n, C, A, B, measure_time);
    return measure_time;
}

double analyze_block(const int n, double *C, const double *A, const double *B, const double *C_expected) {
    double measure_time;
    time_multiplication(multiply_block, n, C, A, B, measure_time);
    check_multiplication_accuracy(C_expected, C, n);
    return measure_time;
}

double analyze_recursive(const int n, double *C, const double *A, const double *B, const double *C_expected) {
    double measure_time;
    time_multiplication(multiply_recursive, n, C, A, B, measure_time);
    check_multiplication_accuracy(C_expected, C, n);
    return measure_time;
}
