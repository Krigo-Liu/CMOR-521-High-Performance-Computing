#include <iostream>
#include <chrono>
#include <vector>

using namespace std;
using namespace std::chrono;

double matsum_pointer(const int n, double *A)
{
    double val = 0.0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            val += A[j + i * n];
        }
    }
    return val;
}

double matsum_VoV(const int n, vector<vector<double> > A)
{
    double val = 0.0;
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            val += A[i][j];
        }
    }
    return val;
}

int main(int argc, char *argv[])
{

    int n = atoi(argv[1]);
    cout << "Matrix size n = " << n << endl;

    double *A = new double[n * n];

    // vector of vectors
    vector< vector<double> > A_VoV;
    for (int i = 0; i < n; ++i)
    {
        vector<double> row;
        for (int j = 0; j < n; ++j)
        {
            row.push_back(0.0);
        }
        A_VoV.push_back(row);
    }

    double min_time = 1e10;
    int num_samples = 25;
    for (int i = 0; i < num_samples; ++i)
    {
        auto start = high_resolution_clock::now();
        double val = matsum_pointer(n, A);
        if (fabs(val) > 1e-11){
            cout << "Something went wrong" << endl;
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        min_time = std::min((double)duration.count(), min_time);
    }
    cout << "Elapsed time for raw pointer array in (microsec): " << min_time << endl;

    min_time = 1e10;
    for (int i = 0; i < num_samples; ++i)
    {
        auto start = high_resolution_clock::now();
        double val = matsum_VoV(n, A_VoV);
        if (fabs(val) > 1e-11){
            cout << "Something went wrong" << endl;
        }
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        min_time = std::min((double)duration.count(), min_time);
    }
    cout << "Elapsed time for vector of vectors in (microsec): " << min_time << endl;

    cout << A[0] << "," << A_VoV[0][0] << endl;

    delete[] A; 
    return 0;
}
