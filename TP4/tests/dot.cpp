#include "utils.hpp"
#include "timer.hpp"
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

double dot(const double* vec0, const double* vec1, const long int N){
   double sum = 0;
   for(long int idx = 0 ; idx < N ; ++idx){
       sum += vec0[idx] * vec1[idx];
   }
   return sum;
}

#if defined(__SSE3__)

#include <emmintrin.h>
#include <pmmintrin.h>


double dot_sse3(const double* vec0, const double* vec1, const long int N){
   
    __m128d sumd = _mm_setzero_pd();

   // Same as N2 = (N - (N%2))
   const long int N2 = (N & ~(long int)1);
   for(long int idx = 0 ; idx < N2 ; idx += 2){
       sumd = _mm_add_pd(sumd, _mm_mul_pd(_mm_loadu_pd(&vec0[idx]), _mm_loadu_pd(&vec1[idx])));
   }
   
    double sum = 0;
    if(N & 1){
        sum += vec0[N-1] * vec1[N-1];
    }
   
   const __m128d res = _mm_add_pd(sumd, _mm_shuffle_pd(sumd, sumd, 1));
   return sum + _mm_cvtsd_f64(res);
}

#endif

#if defined(__AVX2__)

#include <tmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

double hsum(__m256d sumd){
   const __m128d valupper = _mm256_extractf128_pd(sumd, 1);
   const __m128d rest = _mm256_castpd256_pd128(sumd);
   const __m128d valval = _mm_add_pd(valupper, rest);
   const __m128d res    = _mm_add_pd(_mm_permute_pd(valval, 1), valval);
   return _mm_cvtsd_f64(res);
}

double dot_avx2(const double* vec0, const double* vec1, const long int N){
   // every register can hold and do simultanious operation on 4 double precision floating numbers (128 becomes 256)
       __m256d sumd = _mm256_setzero_pd();//change to 256 bits registers

   // Same as N4 = (N - (N%4)) 
   const long int N4 = (N & ~(long int)3); // get the biggest multiple of 4 smaller than N.
   for(long int idx = 0 ; idx < N4 ; idx += 4){
       sumd = _mm256_add_pd(sumd, _mm256_mul_pd(_mm256_loadu_pd(&vec0[idx]), _mm256_loadu_pd(&vec1[idx])));//(128 becomes 256)
   }
   
    double sum = 0;
    if(N & 3){// if N is not multiple of 4
         //  N%4 are left to be calculated seperatly
         for (long int i = N4; i < N; ++i)
        {
            sum += vec0[i] * vec1[i];
        }
    }
   return sum + hsum(sumd);
}

//the difference with dot_avx2 is that loadu becomes load
double dot_avx2_aligned(const double* vec0, const double* vec1, const long int N){
      // every register can hold and do simultanious operation on 4 double precision floating numbers (128 becomes 256)
       __m256d sumd = _mm256_setzero_pd();//change to 256 bits registers

   // Same as N4 = (N - (N%4)) 
   const long int N4 = (N & ~(long int)3); // get the biggest multiple of 4 smaller than N.
   for(long int idx = 0 ; idx < N4 ; idx += 4){

       // i tried this like done in the exercice  8
    //    alignas(64) double v0[4] = {vec0[idx],vec0[idx+1],vec0[idx+2],vec0[idx+3]};
    //    alignas(64) double v1[4] = {vec1[idx],vec1[idx+1],vec1[idx+2],vec1[idx+3]};
    //    sumd = _mm256_add_pd(sumd, _mm256_mul_pd(_mm256_load_pd(v0), _mm256_load_pd(v1)));

       //or simply assuming that the adresses of the vectors are alligned
       sumd = _mm256_add_pd(sumd, _mm256_mul_pd(_mm256_load_pd(&vec0[idx]), _mm256_load_pd(&vec1[idx])));//(128 becomes 256)
   }
   
    double sum = 0;
    if(N & 3){// if N is not multiple of 4
         //  N%4 are left to be calculated seperatly
         for (long int i = N4; i < N; ++i)
        {
            sum += vec0[i] * vec1[i];
        }
    }
   return sum + hsum(sumd);
}

#endif

void test(){
    const long int TestSize = 100000;
    const long int NbLoops = 100;

    std::cout << "Check dot" << std::endl;
    std::cout << "TestSize = " << TestSize << std::endl;

    std::mt19937 gen(0);
    std::uniform_real_distribution<double> dis(0, 1);

    for(long int idx : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 500, 1000, 5000, 10000, 50000}){
        std::cout << "idx = " << idx << std::endl;

        std::vector<double> vec0(idx);
        std::vector<double> vec1(idx);

        double currentSum = 0;
        for(long int idxVal = 0 ; idxVal < idx ; ++idxVal){
            vec0[idxVal] = dis(gen);
            vec1[idxVal] = dis(gen); 
            currentSum += vec0[idxVal]*vec1[idxVal];
        }
        currentSum *= NbLoops;

        double scalarSum = 0;
        {
            Timer timerScalar;

            for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                scalarSum += dot(vec0.data(), vec1.data(), idx);
            }
            timerScalar.stop();

            std::cout << ">> Scalar timer : " << timerScalar.getElapsed() << std::endl;

            CheckEqual(currentSum,scalarSum);
        }
#if defined(__SSE3__)
        {
            double sse3Sum = 0;
            Timer timerSse3;

            for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                sse3Sum += dot_sse3(vec0.data(), vec1.data(), idx);
            }
            timerSse3.stop();

            std::cout << ">> SSE3 timer : " << timerSse3.getElapsed() << std::endl;

            CheckEqual(currentSum,sse3Sum);
        }
#endif
#if defined(__AVX2__)
        {
            double avx2Sum = 0;
            Timer timerAvx2;

            for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                avx2Sum += dot_avx2(vec0.data(), vec1.data(), idx);
            }
            timerAvx2.stop();

            std::cout << ">> AVX2 timer : " << timerAvx2.getElapsed() << std::endl;

            CheckEqual(currentSum,avx2Sum);
        }
#endif
#if defined(__AVX2__)
        {
            double* vec0aligned = reinterpret_cast<double*>(aligned_alloc( 32, sizeof(double) * idx));
            double* vec1aligned = reinterpret_cast<double*>(aligned_alloc( 32, sizeof(double) * idx));

            std::copy(vec0.begin(), vec0.end(), vec0aligned);
            std::copy(vec1.begin(), vec1.end(), vec1aligned);

            double avx2Sum = 0;
            Timer timerAvx2;

            for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
                avx2Sum += dot_avx2_aligned(vec0aligned, vec1aligned, idx);
            }
            timerAvx2.stop();

            std::cout << ">> AVX2 aligned timer : " << timerAvx2.getElapsed() << std::endl;

            CheckEqual(currentSum,avx2Sum);

            free(vec0aligned);
            free(vec1aligned);
        }
#endif
    }
}

int main(){
    test();

    return 0;
}