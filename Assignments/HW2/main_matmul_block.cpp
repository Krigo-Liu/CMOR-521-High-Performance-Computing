#include "./include/cache_block_matmul.hpp"


int STRONG_N    = 2048;  // Fixed matrix dimension for strong scaling
int WEAK_BASE_N = 128;   // Base matrix dimension for 1 thread in weak scaling
int BLOCK_SIZE  = 32;    // Block size for cache blocking

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " [strong|weak]" << endl;
        return 1;
    }
    
    if (strcmp(argv[1], "strong") == 0) {
        run_strong_scaling_experiment();
    } else if (strcmp(argv[1], "weak") == 0) {
        run_weak_scaling_experiment();
    } else {
        cout << "Invalid option. Use 'strong' or 'weak'." << endl;
        return 1;
    }
    
    return 0;
}