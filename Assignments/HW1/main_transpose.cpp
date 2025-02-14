#include "./include/matrix_transpose.hpp"
#include <iostream>

using namespace std;

int BLOCK_SIZE = 16;  
int threshold = 16;   
bool header_written = false;  

int main(int argc, char *argv[])
{
    vector<int> sizes = {32, 64, 128, 256, 512, 1024, 2048};
    vector<int> block_sizes = {8, 16, 32, 64, 128};
    vector<int> thresholds = {8, 16, 32, 64, 128};
    double measure_time;
    string dir_path = "./result";
    if (std::filesystem::exists(dir_path)) {
        std::filesystem::remove_all(dir_path);
    }
    std::filesystem::create_directories(dir_path);

    for (int n : sizes) {
        double *A = new double[n * n];
        double *AT = new double[n * n];

        // Analyze naive transpose
        ofstream naive_file("./result/naive_results.csv", ios::app);
        if(!header_written) {
            naive_file << "Matrix_Size,Time (s)\n";
        }
        measure_time = analyze_naive(n, A, AT);
        naive_file << n << "," << measure_time << "\n";
        naive_file.close();

        
        // Analyze block transpose
        ofstream block_file("./result/block_results.csv", ios::app);
        if(!header_written) {
            block_file << "Matrix_Size,Time (s),Block_Size\n";
        }
        for (int block_size : block_sizes) {
            if (block_size > n) break;
            change_block_size(block_size);
            measure_time = analyze_block(n, A, AT);
            block_file << n << "," << measure_time << "," << BLOCK_SIZE << "\n";
        }
        block_file.close();
        
        // Analyze recursive transpose
        ofstream recursive_file("./result/recursive_results.csv", ios::app);
        if(!header_written) {
            recursive_file << "Matrix_Size,Time (s),Threshold\n";
        }
        for (int threshold : thresholds) {
            if (threshold > n) break;
            change_threshold(threshold);
            measure_time = analyze_recursive(n, A, AT);
            recursive_file << n << "," << measure_time << "," << threshold << "\n";
        }
        recursive_file.close();
        
        delete[] A;
        delete[] AT;

        already_header_written();
    }

    return 0;
}