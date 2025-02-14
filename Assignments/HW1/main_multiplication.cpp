#include "./include/matrix_multiplication.hpp"
#include <iostream>

using namespace std;

int BLOCK_SIZE = 16;  
int threshold = 16;   
bool header_written = false;  

int main(int argc, char *argv[])
{
    // Matrix sizes to test
    vector<int> sizes = {32, 64, 128, 256, 512, 1024};
    // Different block sizes (for block multiplication) and thresholds (for recursive multiplication)
    vector<int> block_sizes = {8, 16, 32, 64};
    vector<int> thresholds  = {8, 16, 32, 64};
    string dir_path = "./result";
    if (std::filesystem::exists(dir_path)) {
        std::filesystem::remove_all(dir_path);
    }
    std::filesystem::create_directories(dir_path);

    for (int n : sizes) {

        // Allocate matrices A, B, C, and a reference matrix C_expected
        double *A  = new double[n * n];
        double *B  = new double[n * n];
        double *C  = new double[n * n];
        double *C_expected = new double[n * n];

        // Randomly initialize A and B
        for (int i = 0; i < n*n; i++) {
            A[i] = rand() / double(RAND_MAX);
            B[i] = rand() / double(RAND_MAX);
        }

        // Analyze naive multiplication
        ofstream naive_file("./result/naive_results_mm.csv", ios::app);
        if (!header_written) {
            naive_file << "Matrix_Size,Time(s)\n";
        }
        double naive_time = analyze_naive(n, C_expected, A, B);
        naive_file << n << "," << naive_time << "\n";
        naive_file.close();

        // Analyze block multiplication
        ofstream block_file("./result/block_results_mm.csv", ios::app);
        if (!header_written) {
            block_file << "Matrix_Size,Time(s),Block_Size\n";
        }
        for (int bs : block_sizes) {
            if (bs > n) break;  // No point in using a block size larger than the matrix
            change_block_size(bs);
            double time_blk = analyze_block(n, C, A, B, C_expected);
            block_file << n << "," << time_blk << "," << BLOCK_SIZE << "\n";
        }
        block_file.close();

        //  Analyze recursive multiplication
        ofstream recursive_file("./result/recursive_results_mm.csv", ios::app);
        if (!header_written) {
            recursive_file << "Matrix_Size,Time(s),Threshold\n";
        }
        for (int thr : thresholds) {
            if (thr > n) break;
            change_threshold(thr);
            double time_rec = analyze_recursive(n, C, A, B, C_expected);
            recursive_file << n << "," << time_rec << "," << threshold << "\n";
        }
        recursive_file.close();

        // Deallocate
        delete[] A;
        delete[] B;
        delete[] C;
        delete[] C_expected;

        // Mark that CSV headers are already written
        already_header_written();
    }

    return 0;
}
