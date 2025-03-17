#include "../include/integral.hpp"

// Function that computes pi using atomic updates
double integral_atomic(int num_steps) {
    double step = 1.0 / num_steps;
    double sum = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        double temp = 4.0 / (1.0 + x * x);
        #pragma omp atomic
        sum += temp;
    }
    return step * sum;
}

// Function that computes pi using the reduction clause
double integral_reduction(int num_steps) {
    double step = 1.0 / num_steps;
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < num_steps; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }
    return step * sum;
}
