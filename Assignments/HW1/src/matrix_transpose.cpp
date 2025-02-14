#include "../include/matrix_transpose.hpp"

using namespace std;



// ------------------------------------------------
// Change the size of BLOCK_SIZE
// ------------------------------------------------
void change_block_size(int block_size) {
    BLOCK_SIZE = block_size;
}

// ------------------------------------------------
// Change the size of BLOCK_SIZE
// ------------------------------------------------
void change_threshold(int threshold_size) {
    threshold = threshold_size;
}

// ------------------------------------------------
// Set the header as written
// ------------------------------------------------
void already_header_written() {
    header_written = true;
}

// ------------------------------------------------
// Function to compute relative error
// ------------------------------------------------
double compute_relative_error(const double * AT_expected, const double * AT_computed, int n) {
    double max_relative_error = 0.0;
    double epsilon = numeric_limits<double>::epsilon();  // Machine precision ~ 1e-16

    for (int i = 0; i < n * n; i++) {
        double abs_error = fabs(AT_expected[i] - AT_computed[i]);
        double denom = fabs(AT_expected[i]) + epsilon;  // Avoid division by zero
        double relative_error = abs_error / denom;

        max_relative_error = max(max_relative_error, relative_error);
    }
    return max_relative_error;
}

// ------------------------------------------------
// Function to check if the relative error is within machine precision
// ------------------------------------------------
void check_transpose_accuracy(const double * AT_expected, const double * AT_computed, int n) {
    double relative_error = compute_relative_error(AT_expected, AT_computed, n);
    
    cout << "Max Relative Error: " << relative_error << endl;

    if (relative_error < numeric_limits<double>::epsilon()) {
        cout << "✅ Transpose implementation is accurate within machine precision.\n";
    } else {
        cout << "❌ WARNING: Transpose implementation has significant numerical errors!\n";
    }
}


// ------------------------------------------------
// Naive Transposition
// ------------------------------------------------
void transpose_naive(const int n, double * AT, double * A) {
    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            AT[i * n + j] = A[j * n + i];
            // AT[j * n + i] = A[i * n + j]; // col major
        }
    }
}

// ------------------------------------------------
// Cache-Blocked Transposition
// -------------------------------------------------

void transpose_block(const int n, double * AT, double * A) {
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int ii = i; ii < i + BLOCK_SIZE && ii < n; ii++) {
                for (int jj = j; jj < j + BLOCK_SIZE && jj < n; jj++) {
                    AT[ii * n + jj] = A[jj * n + ii];
                    // AT[jj * n + ii] = A[ii * n + jj]; // col major
                }
            }
        }
    }
}

// ------------------------------------------------
// Recursive Transposition
// ------------------------------------------------

static void transpose_recursive_helper(const int n, double * AT, double * A, int row, int col, int size, int threshold) {
    if (size <= threshold) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                AT[(col + j) * n + (row + i)] = A[(row + i) * n + (col + j)];
                // AT[(row + i) * n + (col + j)] = A[(col + j) * n + (row + i)]; // col major
            }
        }
    } else {
        int half = size / 2;
        transpose_recursive_helper(n, AT, A, row, col, half, threshold);
        transpose_recursive_helper(n, AT, A, row, col + half, half, threshold);
        transpose_recursive_helper(n, AT, A, row + half, col, half, threshold);
        transpose_recursive_helper(n, AT, A, row + half, col + half, half, threshold);
    }
}

void transpose_recursive(const int n, double * AT, double * A) {
    transpose_recursive_helper(n, AT, A, 0, 0, n, threshold);
}


// ------------------------------------------------
// Timing function (uses std::function)
// ------------------------------------------------
void time_transpose(const std::function<void(const int, double*, double*)> &transpose, const int n, double * AT, double * A, double &time) {
    int trials = 25;
    double min_time = INFINITY;
    clock_t start, end;

    for (int t = 0; t < trials; t++) {
        start = clock();
        transpose(n, AT, A);
        end = clock();

        double elapsed = double(end - start) / CLOCKS_PER_SEC;
        if (elapsed < min_time) {
            min_time = elapsed;
        }
    }
    time = min_time;
}

// ------------------------------------------------
// Analyze naive transposition accross multiple sizes
// Generates a csv file with the results
// ------------------------------------------------
double analyze_naive(const int n, double * AT, double * A) {
    double measure_time;
    time_transpose([&](const int n, double * AT, double * A) {
        transpose_naive(n, AT, A);
    }, n, AT, A, measure_time);
    return measure_time;
    
}

// ------------------------------------------------
// Analyze block transposition accross multiple sizes
// Generates a csv file with the results
// -------------------------------------------------
double analyze_block(const int n, double * AT, double * A) {
    double measure_time;
    double *AT_expected = new double[n * n];

    // Compute correct transposition
    transpose_naive(n, AT_expected, A);

    time_transpose([&](const int n, double * AT, double * A) {
        transpose_block(n, AT, A);
    }, n, AT, A, measure_time);

    // Check accuracy
    check_transpose_accuracy(AT_expected, AT, n);
    return measure_time;
}

// ------------------------------------------------
// Analyze recursive transposition accross multiple sizes
// Generates a csv file with the results
// ------------------------------------------------
double analyze_recursive(const int n, double * AT, double * A) {
    double measure_time;
    double *AT_expected = new double[n * n];

    // Compute correct transposition
    transpose_naive(n, AT_expected, A);

    time_transpose([&](const int n, double * AT, double * A) {
        transpose_recursive(n, AT, A);
    }, n, AT, A, measure_time);

    // Check accuracy
    check_transpose_accuracy(AT_expected, AT, n);
    return measure_time;
}
