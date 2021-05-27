#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <algorithm>

void merge_AB(int A[], const int lengthA,
              const int B[], const int lengthB){
    
    //-----------------------------------------------------------------------------------
    //Please un comment it to evaluate it, I ran out of time so i implemented this version
    //-----------------------------------------------------------------------------------
    // // TODO A and B are sorted
    // // merge B into A while keeping all values sorted
    // int lengthA_real = lengthA-lengthB;
    // int C[lengthA];
    // for(int i=0; i<lengthB;i++){
    //     C[i] = B[i];
    // }
    // for(int i=0; i<lengthA_real;i++){
    //     C[lengthB+i] = A[i];
    // }
    // // TODO allocated size of A is at least (lengthA + lengthB)
    // //I tried this:
    // #std::swap(&A,&C);
    // A = C;
}

int main(int /*argc*/, char** /*argv*/){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 10);
            
    for(int idxSizeA = 0 ; idxSizeA < 100 ; ++idxSizeA){
        for(int idxSizeB = 0 ; idxSizeB < 100 ; ++idxSizeB){
            std::vector<int> A(idxSizeA);
            std::vector<int> B(idxSizeB);
            
            for(int idxA = 0 ; idxA < idxSizeA ; ++idxA){
                if(idxA == 0){
                    A[idxA] = dis(gen);
                }
                else{
                    A[idxA] = dis(gen) + A[idxA-1];
                }
            }
            
            for(int idxB = 0 ; idxB < idxSizeB ; ++idxB){
                if(idxB == 0){
                    B[idxB] = dis(gen);
                    
                }
                else{
                    B[idxB] = dis(gen) + B[idxB-1];
                }
            }
        
            // Not a good solution, merge the array, and sort them....
            std::vector<int> AB = A;
            AB.insert(AB.end(), B.begin(), B.end());
            std::sort(AB.begin(), AB.end());
        
            A.resize(idxSizeA + idxSizeB);
            merge_AB(A.data(), idxSizeA, B.data(), idxSizeB);
            CheckEqual(true, AB == A);
        }
    }
    
    return 0;
}