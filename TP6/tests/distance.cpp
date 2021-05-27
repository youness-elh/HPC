
#include <cuda.h>
#include <cmath>

__global__ void fill(int* blockMat, int* threadMat, const int N){
    const int i_start = blockIdx.y*blockDim.y+threadIdx.y;
    const int j_start = blockIdx.x*blockDim.x+threadIdx.x;

    const int i_inc = gridDim.y*blockDim.y;
    const int j_inc = gridDim.x*blockDim.x;

    for(int i = i_start ; i < N ; i += i_inc){
        for(int j = j_start ; j < N ; j += j_inc){
            blockMat[i*N+j] = blockIdx.y*gridDim.x + blockIdx.x;
            threadMat[i*N+j] = i_start*gridDim.x*blockDim.x + j_start;
        }
    }
}

__global__ void distance(int* distanceMat, const int N){
    const int i_start = blockIdx.y*blockDim.y+threadIdx.y;
    const int j_start = blockIdx.x*blockDim.x+threadIdx.x;

    const int i_inc = gridDim.y*blockDim.y;
    const int j_inc = gridDim.x*blockDim.x;

    for(int i = i_start ; i < N ; i += i_inc){
        for(int j = j_start ; j < N ; j += j_inc){
            distanceMat[i*N+j] = (int) sqrt(pow((i-N/2),2)+pow((j-N/2),2));

        }
    }
}



#include <memory>
#include <iostream>
#include <iomanip>


int main(){
    const int N = 20;


    // Malloc and copy
    int* cuBlockMat;
    cudaMalloc(&cuBlockMat, sizeof(int)*N*N); 
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemset(cuBlockMat, ~0, sizeof(int)*N*N);
    assert(cudaGetLastError() == cudaSuccess);

    int* cuThreadMat;
    cudaMalloc(&cuThreadMat, sizeof(int)*N*N);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemset(cuThreadMat, ~0, sizeof(int)*N*N);
    assert(cudaGetLastError() == cudaSuccess);

    // Create a cuDistance array

    //malloc and copy 
    int* cudistanceMat;
    cudaMalloc(&cudistanceMat, sizeof(int)*N*N); // allocate on the gpu
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemset(cudistanceMat, ~0, sizeof(int)*N*N);// fill it, this is wrong: cuBlockmat[0] = 99
    assert(cudaGetLastError() == cudaSuccess);

    // Call kernel
    CudaCpu(dim3(2,2), dim3(2,2), fill, cuBlockMat, cuThreadMat, N);

    // call my kernel
    CudaCpu(dim3(2,2), dim3(2,2), distance,cudistanceMat, N);

    // Get back results
    std::unique_ptr<int[]> blockMat(new int[N*N]); 
    cudaMemcpy(blockMat.get(), cuBlockMat, sizeof(int)*N*N, cudaMemcpyDeviceToHost);//copying from cpu to gpu
    assert(cudaGetLastError() == cudaSuccess);

    std::unique_ptr<int[]> threadMat(new int[N*N]);
    cudaMemcpy(threadMat.get(), cuThreadMat, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);

    // retreive the results
    std::unique_ptr<int[]> distanceMat(new int[N*N]); //allocate on the cpu 
    cudaMemcpy( distanceMat.get(), cudistanceMat, sizeof(int)*N*N, cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);

    // Free
    cudaFree(cuBlockMat);
    assert(cudaGetLastError() == cudaSuccess);
    cudaFree(cuThreadMat);
    assert(cudaGetLastError() == cudaSuccess);

    // free your array
    cudaFree(cudistanceMat);
    assert(cudaGetLastError() == cudaSuccess);

    // Print result
    std::cout << "blockMat :" << std::endl;
    for(int i = 0 ; i < N ; ++i){
        for(int j = 0 ; j < N ; ++j){
            std::cout << std::setw(3) << blockMat[i*N+j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "threadMat :" << std::endl;
    for(int i = 0 ; i < N ; ++i){
        for(int j = 0 ; j < N ; ++j){
            std::cout << std::setw(3) << threadMat[i*N+j] << " ";
        }
        std::cout << "\n";
    }
    // Print the content of the array
    std::cout << "distanceMat :" << std::endl;
    for(int i = 0 ; i < N ; ++i){
        for(int j = 0 ; j < N ; ++j){
            std::cout << std::setw(3) << distanceMat[i*N+j] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}