#include "../include/matrix_transpose.hpp"

using namespace std;

// -----------------------------------------------------
// Naive Transposition
// -----------------------------------------------------
void transpose_naive(const int n, double *AT, double *A) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            AT[j * n + i] = A[i * n + j];
        }
    }
}

// -----------------------------------------------------
// Cache-Blocked Transposition
// -----------------------------------------------------
void transpose_blocked(const int n, double *AT, double *A, int block_size) {
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int bi = i; bi < i + block_size && bi < n; bi++) {
                for (int bj = j; bj < j + block_size && bj < n; bj++) {
                    AT[bj * n + bi] = A[bi * n + bj];
                }
            }
        }
    }
}

// -----------------------------------------------------
// Recursive Transposition
// -----------------------------------------------------
static void transpose_recursive_helper(const int n, double *AT, double *A,
                                       int row, int col, int size, int threshold) {
    if (size <= threshold) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                AT[(col + j) * n + (row + i)] = A[(row + i) * n + (col + j)];
            }
        }
    } else {
        int half = size / 2;
        transpose_recursive_helper(n, AT, A, row,      col,      half, threshold);
        transpose_recursive_helper(n, AT, A, row,      col+half, half, threshold);
        transpose_recursive_helper(n, AT, A, row+half, col,      half, threshold);
        transpose_recursive_helper(n, AT, A, row+half, col+half, half, threshold);
    }
}

void transpose_recursive(const int n, double *AT, double *A, int threshold) {
    transpose_recursive_helper(n, AT, A, 0, 0, n, threshold);
}

// -----------------------------------------------------
// Timing function (uses std::function)
// -----------------------------------------------------
void time_transpose(const std::function<void(const int, double*, double*)> &transpose,
                    const int n, double *AT, double *A, double &time) {
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

// -----------------------------------------------------
// Analyze naive transpose across multiple sizes
// Generates naive_results.csv
// -----------------------------------------------------
void analyze_naive() {
    vector<int> sizes = {256, 512, 768, 1024, 2048};
    ofstream file("naive_results.csv");
    file << "Matrix_Size,Time (s)\n";

    for (int curr_n : sizes) {
        // Allocate for each test
        vector<double> A(curr_n * curr_n), AT(curr_n * curr_n);

        // Initialize A
        for (int i = 0; i < curr_n * curr_n; i++) {
            A[i] = static_cast<double>(rand()) / RAND_MAX;
        }

        double measured_time;
        time_transpose(transpose_naive, curr_n, AT.data(), A.data(), measured_time);
        file << curr_n << "," << measured_time << "\n";
    }

    file.close();
}


// -----------------------------------------------------
// Analyze different block sizes
// -----------------------------------------------------
void analyze_block_sizes(int n, vector<double> &A, vector<double> &AT) {
    vector<int> block_sizes = {8, 16, 32, 64, 128};
    ofstream file("block_sizes_results.csv");
    file << "Block_Size,Time (s)\n";

    for (int block_size : block_sizes) {
        double best_time;
        time_transpose(
            [&](const int n, double *dest, double *src) {
                transpose_blocked(n, dest, src, block_size);
            },
            n,
            AT.data(),
            A.data(),
            best_time
        );

        file << block_size << "," << best_time << "\n";
    }
    file.close();
}

// -----------------------------------------------------
// Analyze different recursion thresholds
// -----------------------------------------------------
void analyze_threshold_sizes(int n, vector<double> &A, vector<double> &AT) {
    vector<int> thresholds = {16, 32, 64, 128, 256};
    ofstream file("threshold_sizes_results.csv");
    file << "Threshold_Size,Time (s)\n";

    for (int threshold : thresholds) {
        double best_time;
        time_transpose(
            [&](const int n, double *dest, double *src) {
                transpose_recursive(n, dest, src, threshold);
            },
            n,
            AT.data(),
            A.data(),
            best_time
        );

        file << threshold << "," << best_time << "\n";
    }
    file.close();
}

// -----------------------------------------------------
// main()
// -----------------------------------------------------
int main() {
    int n = 1024; // Matrix size

    // Allocate using std::vector
    vector<double> A(n * n);
    vector<double> AT(n * n);

    // Initialize matrix
    for (int i = 0; i < n * n; i++) {
        A[i] = static_cast<double>(rand()) / RAND_MAX;
    }

    // Analyze block sizes
    analyze_block_sizes(n, A, AT);

    // Analyze recursion thresholds
    analyze_threshold_sizes(n, A, AT);

    // Analyze naive transpose
    analyze_naive();

    return 0;
}
