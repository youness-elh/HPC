 
#include <cuda.h>
#include <limits>
#include <math.h>  

__global__ void sumThemAll(const long int* valuesToSum, long int* currentSum, const long int N){
    // if thread id is less than N, sum the values
    const int i_start = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_inc = gridDim.x * blockDim.x;
    long int sum = 0;
    for (int i = i_start; i < N; i+=i_inc)
    {
        sum += *(valuesToSum + i);
    }

    *(currentSum + i_start) = sum; 
}


#include <memory>
#include <iostream>

int main(){
    const int MaxThreadsPerGroup = 10;
    const int MaxGroup = 5;
    const int N = 1000;

    std::unique_ptr<long int[]> values(new long int[N]);

    for(int i = 0 ; i < N ; ++i){
        values[i] = i;
    }
    const long int expectedSum = (N-1)*(N-1+1)/2;

    //long int *finalSum= new long int [N];
    long int finalSum = -1;
    
    // Malloc and copy
    long int* cuValues;
    cudaMalloc(&cuValues, sizeof(long int)*N);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemcpy(cuValues, values.get(), sizeof(long int)*N, cudaMemcpyHostToDevice);
    assert(cudaGetLastError() == cudaSuccess);

    long int* cuBuffer;
    cudaMalloc(&cuBuffer, sizeof(long int)*N);
    assert(cudaGetLastError() == cudaSuccess);
    cudaMemset(cuBuffer, 0, sizeof(long int)*N);
    assert(cudaGetLastError() == cudaSuccess);

    // loop to sum the array multiple times until there is only one value
    // Call kernel

    long int size= N;
    int newMaxGroupp = MaxGroup;
    int newMaxThreadsPerGroup = MaxThreadsPerGroup;
    const int test_size = 2*MaxThreadsPerGroup;

    while(size >= 2){
        CudaCpu(dim3(newMaxGroupp), dim3(newMaxThreadsPerGroup), sumThemAll, cuValues, cuBuffer, size);
        //std::cout << cuBuffer[0];
        size = newMaxGroupp*newMaxThreadsPerGroup;

        newMaxGroupp = (size >= test_size)? size % test_size + int(size/test_size) : 1;
        newMaxThreadsPerGroup = (size >= test_size)?  MaxThreadsPerGroup : int(size/2) ;
 
        std::swap(cuBuffer,cuValues);

    }
    //Get back results
    cudaMemcpy(&finalSum, cuValues, sizeof(long int), cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);

    // Free
    cudaFree(cuValues);
    assert(cudaGetLastError() == cudaSuccess);
    cudaFree(cuBuffer);
    assert(cudaGetLastError() == cudaSuccess);

    
    if(finalSum == expectedSum){
        std::cout << "Correct! Sum found is : " << finalSum << std::endl;    
    }
    else{
        std::cout << "Error! Sum found is : " << finalSum << " should be " << expectedSum << std::endl;       
    }

    return 0;
}