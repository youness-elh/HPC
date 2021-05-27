#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <omp.h>
#include <math.h>  


int Fibonacci(int n) {
  if (n < 2)
    return n;
  else
    return Fibonacci(n-1) + Fibonacci(n-2);
}
//decreasing function putting more weight on the top level of the recurssion (hight value of n implies 0 first priority 0)
int f(int n) {
    return exp(-n);
}

int FibonacciOmp(int n) {

    // The master thread will do this
    int f1, f2;
    if(n<2)
        return n;
    {
        if(n<15){
            return FibonacciOmp(n-1)+FibonacciOmp(n-2);
        }
            
    // Create a task that can be executed by any threads
    //#pragma omp task shared(f2,n)
    //Use higher priority for tasks at the top (less recursion level)
    #pragma omp task priority(f(n)) shared(f2,n) 
    f2 = FibonacciOmp(n-2);

    // Create a task that can be executed by any threads
    //#pragma omp task shared(f1,n)
    //this is quicker than creating a task to another thread, the master is waiting for tasks to be done anyhow so it is efficient if he is also doing a task in the meantime
    f1 = FibonacciOmp(n-1);

    #pragma omp taskwait
    }
    return f1+f2;
}

void test(){
    const long int TestSize = 40;
    const long int NbLoops = 10;

    std::cout << "Check Fibonacci" << std::endl;
    std::cout << "TestSize = " << TestSize << std::endl;

    int scalarFibonnaci = 0;
    {
        Timer timerSequential;

        for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            scalarFibonnaci += Fibonacci(TestSize);
        }
        timerSequential.stop();

        std::cout << ">> Sequential timer : " << timerSequential.getElapsed() << std::endl;
    }
    #pragma omp parallel
    #pragma omp master
    {

        int ompFibonnaci = 0;
        Timer timerParallel;

        for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            ompFibonnaci += FibonacciOmp(TestSize);
        }

        timerParallel.stop();

        std::cout << ">> There are " << omp_get_num_threads() << " threads" << std::endl;
        std::cout << ">> Omp timer : " << timerParallel.getElapsed() << std::endl;

        CheckEqual(scalarFibonnaci,ompFibonnaci);
    }
}

int main(){
    test();

    return 0;
}