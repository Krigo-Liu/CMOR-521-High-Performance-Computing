#include <omp.h>
#include <iostream>

using namespace std;

#define NUM_THREADS 10

int main(){
    int num_steps = 100000000;
    double step = 1.0 / (double) num_steps;
    for (int i = 1; i <= NUM_THREADS; ++i){
        omp_set_num_threads(i);
        double pi = 0.0;
        double sum = 0.0;
        double elapsed_time = omp_get_wtime();
        #pragma omp parallel for reduction(+:sum)
            for (int j = 0; j < num_steps; ++j){
                double x = (j + 0.5) * step;
                sum += 4.0 / (1.0 + x * x);
            }
        pi += step * sum;
    elapsed_time = omp_get_wtime() - elapsed_time;
    cout << "cores = " << i << " " << "pi = " << pi << " in " << elapsed_time << " secs \n" << endl;
    }

}