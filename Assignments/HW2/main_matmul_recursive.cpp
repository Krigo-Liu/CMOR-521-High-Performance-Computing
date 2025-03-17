#include "./include/recursive_matmul.hpp"

int STRONG_MATRIX_SIZE = 4098; 
int WEAK_BASE_MATRIX_SIZE = 512;
int MIN_RECURSIVE_SIZE = 32;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " [strong|weak]" << endl;
        return 1;
    }

    if (strcmp(argv[1], "strong") == 0) {
        runStrongScalingExperiment();
    } else if (strcmp(argv[1], "weak") == 0) {
        runWeakScalingExperiment();
    } else {
        cout << "Invalid option. Use 'strong' or 'weak'." << endl;
        return 1;
    }

    return 0;
}