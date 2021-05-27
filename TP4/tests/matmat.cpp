#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>

// A classic matrix matrix product of size 4
void matmat4x4(double C[4][4], const double A[4][4], const double B[4][4]){
    for(int i = 0 ; i < 4 ; ++i){
        for(int j = 0 ; j < 4 ; ++j){
            for(int k = 0 ; k < 4 ; ++k){
                C[i][j] += A[i][k] * B[j][k];
            }
        }
    }
}

#ifndef __AVX2__
#error AVX2 must be enabled
#endif

#include <tmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

// This sum all the values in vector sumd into a scalar
double hsum(__m256d sumd){
   const __m128d valupper = _mm256_extractf128_pd(sumd, 1);
   const __m128d rest = _mm256_castpd256_pd128(sumd);
   const __m128d valval = _mm_add_pd(valupper, rest);
   const __m128d res    = _mm_add_pd(_mm_permute_pd(valval, 1), valval);
   return _mm_cvtsd_f64(res);
}

// This sum all the values in 4 vectors into a scalar
__m256d h4sum(__m256d sumd0, __m256d sumd1, __m256d sumd2, __m256d sumd3){
   const __m128d valupper0 = _mm256_extractf128_pd(sumd0, 1);
   const __m128d rest0 = _mm256_castpd256_pd128(sumd0);
      
   const __m128d valupper1 = _mm256_extractf128_pd(sumd1, 1);
   const __m128d rest1 = _mm256_castpd256_pd128(sumd1);
      
   const __m128d valupper2 = _mm256_extractf128_pd(sumd2, 1);
   const __m128d rest2 = _mm256_castpd256_pd128(sumd2);   
      
   const __m128d valupper3 = _mm256_extractf128_pd(sumd3, 1);
   const __m128d rest3 = _mm256_castpd256_pd128(sumd3);
      
   const __m256d sumd02 = _mm256_insertf128_pd(_mm256_castpd128_pd256(valupper0),valupper2,1);
   const __m256d sumd02bis = _mm256_insertf128_pd(_mm256_castpd128_pd256(rest0),rest2,1);
   
   const __m256d sumd13 = _mm256_insertf128_pd(_mm256_castpd128_pd256(valupper1),valupper3,1);
   const __m256d sumd13bis = _mm256_insertf128_pd(_mm256_castpd128_pd256(rest1),rest3,1);
        
   const __m256d sumd0202bis = _mm256_add_pd(sumd02, sumd02bis);
   const __m256d sumd1313bis = _mm256_add_pd(sumd13, sumd13bis);
   
   return _mm256_hadd_pd (sumd0202bis, sumd1313bis);
}

void matmat4x4_avx(double C[4][4], const double A[4][4], const double B[4][4]){
    // multiply line by line (B = B.T)
    __m256d sumd0 = _mm256_setzero_pd();
    __m256d sumd1 = _mm256_setzero_pd();
    __m256d sumd2 = _mm256_setzero_pd();
    __m256d sumd3 = _mm256_setzero_pd();

    //for each line of A 
    for(long int idx = 0 ; idx < 4 ; idx++){
        // for every column/here line of B
        sumd0 = _mm256_mul_pd(_mm256_load_pd(A[idx]), _mm256_load_pd(B[0]));
        sumd1 = _mm256_mul_pd(_mm256_load_pd(A[idx]), _mm256_load_pd(B[1]));
        sumd2 =  _mm256_mul_pd(_mm256_load_pd(A[idx]), _mm256_load_pd(B[2]));
        sumd3 =  _mm256_mul_pd(_mm256_load_pd(A[idx]), _mm256_load_pd(B[3]));
    
        //load the output line      
        __m256d  C_line = _mm256_load_pd(C[idx]);
        //obtain the first line of c
        C_line += h4sum(sumd0, sumd1, sumd2, sumd3);
        //store the result
         _mm256_store_pd(&C[idx][0],C_line);
    }
}

void test(){
    const long int NbLoops = 10000;

    std::cout << "Check matmat4x4" << std::endl;

    alignas(32) double A[4][4];
    alignas(32) double B[4][4];
    {
        std::mt19937 gen(0);
        std::uniform_real_distribution<double> dis(0, 1);

        for(int i = 0 ; i < 4 ; ++i){
            for(int j = 0 ; j < 4 ; ++j){
                A[i][j] = dis(gen);
                B[j][i] = dis(gen);
            }
        }
    }

    alignas(32) double C[4][4] = {0};
    {
        Timer timerScalar;

        for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            matmat4x4(C, A, B);
        }
        timerScalar.stop();

        std::cout << ">> Scalar timer : " << timerScalar.getElapsed() << std::endl;

    }

    alignas(32) double Cavx[4][4] = {0};
    {
        Timer timerAvx2;

        for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
            matmat4x4_avx(Cavx, A, B);
        }
        timerAvx2.stop();

        std::cout << ">> AVX2 timer : " << timerAvx2.getElapsed() << std::endl;
    }

    for(int i = 0 ; i < 4 ; ++i){
        for(int j = 0 ; j < 4 ; ++j){
            //std::cout << "------check--------" << Cavx[i][j] << std::endl;
            CheckEqual(C[i][j],Cavx[i][j]);
        }
    }
}

int main(){
    test();

    return 0;
}