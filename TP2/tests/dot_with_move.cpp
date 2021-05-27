#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <sched.h>

#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>

long int GetBinding(){
    cpu_set_t mask;
    CPU_ZERO(&mask);
    pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
    // Get the affinity
    int retValue = sched_getaffinity(tid, sizeof(mask), &mask);
    assert(retValue == 0);
    long int retMask = 0;
    for(size_t idx = 0 ; idx < sizeof(long int)*8-1 ; ++idx){
        if(CPU_ISSET(idx, &mask)){
            retMask |= (1<<idx);
        }
    }
    return retMask;
}

void BindToCore(const int inCoreId){
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(inCoreId, &set);

    pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
    int retValue = sched_setaffinity(tid, sizeof(set), &set);
    assert(retValue == 0);
}

std::vector<long int> GetBindingList(){
    const long int cores = GetBinding();
    
    std::vector<long int> list;
    long int idx = 0;
    while((1 << idx) <= cores){
        if((1 << idx) & cores){
             list.push_back(idx);
        }
        idx += 1;
    }
    
    return list;
}


double dot(const double* vec0, const double* vec1, const long int N){
   double sum = 0;
   for(long int idx = 0 ; idx < N ; ++idx){
       sum += vec0[idx] * vec1[idx];
   }
   return sum;
}

void test(){
    const std::vector<long int> availableCores = GetBindingList();
    BindToCore(int(availableCores[0]));

    const long int TestSize = 500000;
    const long int NbLoops = 1000;

    std::cout << "Check dot" << std::endl;
    std::cout << "TestSize = " << TestSize << std::endl;

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dis(0, 1);

    std::vector<double> vec0(TestSize);
    std::vector<double> vec1(TestSize);

    double currentSum = 0;
    for(long int idxVal = 0 ; idxVal < TestSize ; ++idxVal){
        vec0[idxVal] = dis(gen);
        vec1[idxVal] = dis(gen);
        currentSum += vec0[idxVal]*vec1[idxVal];
    }
    currentSum *= NbLoops;

    {
        double scalarSum = 0;
        Timer timerScalar;

        for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            scalarSum += dot(vec0.data(), vec1.data(), TestSize);
        }
        timerScalar.stop();

        std::cout << ">> Without move : " << timerScalar.getElapsed() << std::endl;

        CheckEqual(currentSum,scalarSum);
    }
    {
    
        double scalarWithStopSum = 0;
        Timer timerWithStop;

        for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            scalarWithStopSum += dot(vec0.data(), vec1.data(), TestSize);
            //added line
            //std::cout << "size of av cores" << availableCores.size() << std::endl;
            int nb_cores = int(availableCores.size());
            BindToCore((sched_getcpu()+1)%nb_cores);
        }
        timerWithStop.stop();

        std::cout << ">> With move : " << timerWithStop.getElapsed() << std::endl;

        CheckEqual(currentSum,scalarWithStopSum);
    }
}

int main(){
    test();

    return 0;
}
