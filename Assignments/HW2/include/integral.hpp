#ifndef INTEGRAL_HPP
#define INTEGRAL_HPP

#include <omp.h>
#include <iostream>
#include <iomanip>

using namespace std;

double integral_atomic(int num_steps);
double integral_reduction(int num_steps);

#endif // INTEGRAL_HPP