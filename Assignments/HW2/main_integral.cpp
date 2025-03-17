#include "./include/integral.hpp"

int main() {
    int num_steps = 100000000; // fixed problem size
    int thread_counts[] = {1, 2, 4, 8, 16};
    int num_tests = sizeof(thread_counts) / sizeof(thread_counts[0]);
    
    double base_time_atomic = 0.0;
    double base_time_reduction = 0.0;
    
    cout << fixed << setprecision(6);
    cout << "Strong Scaling Study for PI Integration (num_steps = " << num_steps << ")\n\n";
    
    // Study for the atomic-based implementation
    cout << "Atomic Implementation:\n";
    cout << "Threads\tTime(s)\tSpeedup\tEfficiency\tPi\n";
    for (int i = 0; i < num_tests; i++) {
        omp_set_num_threads(thread_counts[i]);
        double start = omp_get_wtime();
        double pi = integral_atomic(num_steps);
        double elapsed = omp_get_wtime() - start;
        if(i == 0)
            base_time_atomic = elapsed;
        double speedup = base_time_atomic / elapsed;
        double efficiency = speedup / thread_counts[i];
        cout << thread_counts[i] << "\t" << elapsed << "\t" 
             << speedup << "\t" << efficiency << "\t\t" << pi << "\n";
    }
    
    cout << "\n";
    
    // Study for the reduction-based implementation
    cout << "Reduction Implementation:\n";
    cout << "Threads\tTime(s)\tSpeedup\tEfficiency\tPi\n";
    for (int i = 0; i < num_tests; i++) {
        omp_set_num_threads(thread_counts[i]);
        double start = omp_get_wtime();
        double pi = integral_reduction(num_steps);
        double elapsed = omp_get_wtime() - start;
        if(i == 0)
            base_time_reduction = elapsed;
        double speedup = base_time_reduction / elapsed;
        double efficiency = speedup / thread_counts[i];
        cout << thread_counts[i] << "\t" << elapsed << "\t" 
             << speedup << "\t" << efficiency << "\t\t" << pi << "\n";
    }
    
    return 0;
}