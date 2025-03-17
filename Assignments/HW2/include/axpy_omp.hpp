#ifndef AXPYOMP_HPP
#define AXPYOMP_HPP

#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

using namespace std;

void axpy_version_1(int n, double *y, double alpha, double *x);
void axpy_version_1(int n, double *y, double alpha, double *x, int num_threads);

double axpy_version_1_return(int n, double *y, double alpha, double *x);
double axpy_version_1_return(int n, double *y, double alpha, double *x, int num_threads);


void axpy_version_2(int n, double *y, double alpha, double *x);
void axpy_version_2(int n, double *y, double alpha, double *x, int num_threads);

double axpy_version_2_return(int n, double *y, double alpha, double *x);
double axpy_version_2_return(int n, double *y, double alpha, double *x, int num_threads);

#endif // AXPYOMP_HPP