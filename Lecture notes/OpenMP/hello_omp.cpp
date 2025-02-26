#include <omp.h>
#include <iostream>

using namespace std;

int main()
{
  omp_set_num_threads(1);

#pragma omp parallel 
  {
    int tid = omp_get_thread_num();
    cout << "Hello world from thread " << 
      tid << " / " << omp_get_num_threads() 
      << " threads" << endl;
  }

#pragma omp parallel num_threads(2)
  {
    int tid = omp_get_thread_num();
    cout << "Hello world again from thread " << 
      tid << " / " << omp_get_num_threads() 
      << " threads" << endl;
  }  
  return 0;
}
