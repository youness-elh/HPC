#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <omp.h>
#include <memory>
#include <cstring>
#include <array>

//#define SOLUTION


#define PARALLELIZE
//#define VECTORIZE


int main(int argc, char** argv){
    if(argc != 2 || (argc == 2 && (strcmp(argv[1], "-check") !=0 && strcmp(argv[1], "-no-check") && strcmp(argv[1], "-no-optim")))){
        std::cout << "Should be\n";
        std::cout << argv[0] << " -check #to check the result\n";
        std::cout << argv[0] << " -no-check #to avoid checking the result\n";   
        std::cout << argv[0] << " -no-optim #to avoid running the optimized code\n";  
        return 1;     
    }

    const bool checkRes = (strcmp(argv[1], "-no-check") != 0);
    const bool runOptim = (strcmp(argv[1], "-no-optim") != 0);

    const long int N = 1024;
    double* A = (double*)aligned_alloc(64, N * N * sizeof(double));
    memset(A, 0, N * N * sizeof(double));
    double* B = (double*)aligned_alloc(64, N * N * sizeof(double));
    memset(B, 0, N * N * sizeof(double));
    double* C = (double*)aligned_alloc(64, N * N * sizeof(double));
    memset(C, 0, N * N * sizeof(double));
    double* COptim = (double*)aligned_alloc(64, N * N * sizeof(double));
    memset(COptim, 0, N * N * sizeof(double));
   
    {        
        std::mt19937 gen(0);
        std::uniform_real_distribution<double> dis(0, 1);
        
        for(long int i = 0 ; i < N ; ++i){
            for(long int j = 0 ; j < N ; ++j){
                A[i*N+j] = dis(gen);
                B[j*N+i] = dis(gen);
            }
        }
    }   
    
    Timer timerNoOptim;
    if(checkRes){
        for(long int k = 0 ; k < N ; ++k){
            for(long int j = 0 ; j < N ; ++j){
                for(long int i = 0 ; i < N ; ++i){
                    C[i*N+j] += A[i*N+k] * B[j*N+k];
                }
            }
        }
    }
    timerNoOptim.stop();
    
    Timer timerWithOptim;
    if(runOptim){
    // 3 Improvemnets

    #ifdef SOLUTION
    // reversed loop order :
      for(long int i = 0 ; i < N ; ++i){
          for(long int j = 0 ; j < N ; ++j){
              for(long int k = 0 ; k < N ; ++k){
                  COptim[i*N+j] += A[i*N+k] * B[j*N+k];
              }
          }
      }
    #endif
    
    
    #ifdef PARALLELIZE
      // parallelization :
      double tmp;
      for(long int i = 0 ; i < N ; ++i){
          for(long int j = 0 ; j < N ; ++j){
            tmp = 0;
            #pragma omp parallel for reduction(+ : tmp)
              for(long int k = 0 ; k < N ; ++k){
                tmp  += A[i*N+k] * B[j*N+k];
              }
              COptim[i*N+j] += tmp;
          }
      }
    #endif
    
    #ifdef VECTORIZE
    	double tmp;
      #pragma omp parallel for
    	for(long int i = 0 ; i < N ; ++i){
            for(long int j = 0 ; j < N ; ++j){
            	tmp = 0;
                double* bufA = &A[i*N];
                double* bufB = &B[j*N];
                #pragma omp simd reduction(+ : tmp) safelen(N)
                for(long int k = 0 ; k < N ; ++k){
                    tmp += bufA[k] * bufB[k];
                }
                COptim[i*N+j] += tmp;
            }
        }
      #endif
    
    }
    timerWithOptim.stop();
    
    if(checkRes){
        std::cout << ">> Without Optim : " << timerNoOptim.getElapsed() << std::endl;
        if(runOptim){
            for(long int i = 0 ; i < N ; ++i){
                for(long int j = 0 ; j < N ; ++j){
                    CheckEqual(C[i*N+j],COptim[i*N+j]);
                }
            }
        }   
    }
    if(runOptim){
        std::cout << ">> With Optim : " << timerWithOptim.getElapsed() << std::endl;
    }
    
    free(A);
    free(B);
    free(C);
    free(COptim);
    
    return 0;
}