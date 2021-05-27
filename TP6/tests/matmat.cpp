
#include <cuda.h>
#include <cmath>
#include <iomanip>

__global__ void matmat(float* A, float* B, float* C, const int N){

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            A[ROW * N + COL] += B[ROW * N + i] * C[i * N + COL];
        }
    }
  
}


#include <memory>
#include <iostream>

int main(){
    const int N = 100;

    std::unique_ptr<float[]> A(new float[N*N]);
    std::unique_ptr<float[]> B(new float[N*N]);
    std::unique_ptr<float[]> C(new float[N*N]);

    for(int i = 0 ; i < N ; ++i){
        for(int j = 0 ; j < N ; ++j){
            B[i*N+j] = float(i * j);
            C[i*N+j] = float(i + j);
        }
    }

    for(int i = 0 ; i < N ; ++i){
        for(int j = 0 ; j < N ; ++j){
            for( int k = 0 ; k < N ; ++k){
               A[i*N+j] += B[i*N+k] * C[k*N+j];
            }
        }
    }

    std::unique_ptr<float[]> A_from_CUDA(new float[N*N]());
    // allocate cuA cuB and cuC
    //cuA
    float* cuA;
    cudaMalloc(&cuA, sizeof(int)*N*N); 
    assert(cudaGetLastError() == cudaSuccess);

    //cuB
    float* cuB;
    cudaMalloc(&cuB, sizeof(int)*N*N); 
    assert(cudaGetLastError() == cudaSuccess);
    //cudaMemset(cuA, 0, sizeof(int)*N*N); 
    //assert(cudaGetLastError() == cudaSuccess);

    //cuC
    float* cuC;
    cudaMalloc(&cuC, sizeof(int)*N*N); 
    assert(cudaGetLastError() == cudaSuccess);
    //cudaMemset(cuA, 0, sizeof(int)*N*N); 
    //assert(cudaGetLastError() == cudaSuccess);
    
    // copy B and C to cuB and cuC //from cpu to gpu cudaMemcpy(dest,src,...)
    cudaMemcpy( cuB, B.get(), sizeof(int)*N*N, cudaMemcpyHostToDevice);
    assert(cudaGetLastError() == cudaSuccess);

    cudaMemcpy( cuC, C.get(), sizeof(int)*N*N, cudaMemcpyHostToDevice);
    assert(cudaGetLastError() == cudaSuccess);

    // initi cuA to zero
    cudaMemset(cuA, ~0, sizeof(int)*N*N); 
    assert(cudaGetLastError() == cudaSuccess);
    
    // call your kernel
    CudaCpu(dim3(N,N), dim3(1,1), matmat, cuA, cuB, cuC, N);

    // copy back your result from cuA into A_from_CUDA
    cudaMemcpy( A_from_CUDA.get(), cuA, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);

    // Check result
    float error = 0;
    for(int i = 0 ; i < N ; ++i){
        for(int j = 0 ; j < N ; ++j){
            error = std::max(error, (A[i*N+j] == 0 ? A_from_CUDA[i*N+j] : std::abs((A_from_CUDA[i*N+j]-A[i*N+j])/A[i*N+j])));
        }
    }

    std::cout << "Error is : " << error << std::endl;

    // Print the content of the array
    // std::cout << "A_from_CUDA :" << std::endl;
    // for(int i = 0 ; i < 6; ++i){
    //     for(int j = 0 ; j < 6 ; ++j){
    //         std::cout << std::setw(3) << A_from_CUDA[i*N+j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    // // Print the content of the array
    // std::cout << "A :" << std::endl;
    // for(int i = 0 ; i < 6 ; ++i){
    //     for(int j = 0 ; j < 6 ; ++j){
    //         std::cout << std::setw(3) << A[i*N+j] << " ";
    //     }
    //     std::cout << "\n";
    // }

    return 0;
}