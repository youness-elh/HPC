#ifndef CUCACPU_MEM_H
#define CUCACPU_MEM_H

#include "cudacpu_syntax.h"
#include "cudacpu_error.h"

#include <cstring>
#include <cstdlib>

enum cudaMemcpyKind {
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice
};


__host__ __device__ cudaError_t cudaFree ( void* devPtr ){
    std::free(devPtr);
    return cudaSuccess;
}

__host__ cudaError_t cudaFreeHost ( void* ptr ){
    std::free(ptr);
    return cudaSuccess;
}

template <class ArrayType>
__host__ __device__ cudaError_t cudaMalloc ( ArrayType** devPtr, size_t size ){
    (*devPtr) = (ArrayType*)std::malloc(size);
    if((*devPtr)) return cudaSuccess;
    else return cudaErrorMemoryAllocation;
}

__host__ cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind /*kind*/ ){
    memcpy(dst, src, count);
    return cudaSuccess;
}

__host__ cudaError_t cudaMemset ( void* devPtr, int  value, size_t count ){
    memset(devPtr, value, count);
    return cudaSuccess;
}


/*
TODO:
__host__ ​cudaError_t cudaFreeArray ( cudaArray_t array )
__host__ ​cudaError_t cudaArrayGetInfo ( cudaChannelFormatDesc* desc, cudaExtent* extent, unsigned int* flags, cudaArray_t array )
    Gets info about the specified cudaArray.
    Frees memory on the device.
    Frees an array on the device.
__host__ ​cudaError_t cudaFreeMipmappedArray ( cudaMipmappedArray_t mipmappedArray )
    Frees a mipmapped array on the device.
__host__ ​cudaError_t cudaGetMipmappedArrayLevel ( cudaArray_t* levelArray, cudaMipmappedArray_const_t mipmappedArray, unsigned int  level )
    Gets a mipmap level of a CUDA mipmapped array.
__host__ ​cudaError_t cudaGetSymbolAddress ( void** devPtr, const void* symbol )
    Finds the address associated with a CUDA symbol.
__host__ ​cudaError_t cudaGetSymbolSize ( size_t* size, const void* symbol )
    Finds the size of the object associated with a CUDA symbol.
__host__ ​cudaError_t cudaHostAlloc ( void** pHost, size_t size, unsigned int  flags )
    Allocates page-locked memory on the host.
__host__ ​cudaError_t cudaHostGetDevicePointer ( void** pDevice, void* pHost, unsigned int  flags )
    Passes back device pointer of mapped host memory allocated by cudaHostAlloc or registered by cudaHostRegister.
__host__ ​cudaError_t cudaHostGetFlags ( unsigned int* pFlags, void* pHost )
    Passes back flags used to allocate pinned host memory allocated by cudaHostAlloc.
__host__ ​cudaError_t cudaHostRegister ( void* ptr, size_t size, unsigned int  flags )
    Registers an existing host memory range for use by CUDA.
__host__ ​cudaError_t cudaHostUnregister ( void* ptr )
    Unregisters a memory range that was registered with cudaHostRegister.

__host__ ​cudaError_t cudaMalloc3D ( cudaPitchedPtr* pitchedDevPtr, cudaExtent extent )
    Allocates logical 1D, 2D, or 3D memory objects on the device.
__host__ ​cudaError_t cudaMalloc3DArray ( cudaArray_t* array, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  flags = 0 )
    Allocate an array on the device.
__host__ ​cudaError_t cudaMallocArray ( cudaArray_t* array, const cudaChannelFormatDesc* desc, size_t width, size_t height = 0, unsigned int  flags = 0 )
    Allocate an array on the device.
__host__ ​cudaError_t cudaMallocHost ( void** ptr, size_t size )
    Allocates page-locked memory on the host.
__host__ ​cudaError_t cudaMallocManaged ( void** devPtr, size_t size, unsigned int  flags = cudaMemAttachGlobal )
    Allocates memory that will be automatically managed by the Unified Memory system.
__host__ ​cudaError_t cudaMallocMipmappedArray ( cudaMipmappedArray_t* mipmappedArray, const cudaChannelFormatDesc* desc, cudaExtent extent, unsigned int  numLevels, unsigned int  flags = 0 )
    Allocate a mipmapped array on the device.
__host__ ​cudaError_t cudaMallocPitch ( void** devPtr, size_t* pitch, size_t width, size_t height )
    Allocates pitched memory on the device.
__host__ ​cudaError_t cudaMemAdvise ( const void* devPtr, size_t count, cudaMemoryAdvise advice, int  device )
    Advise about the usage of a given memory range.
__host__ ​cudaError_t cudaMemGetInfo ( size_t* free, size_t* total )
    Gets free and total device memory.
__host__ ​cudaError_t cudaMemPrefetchAsync ( const void* devPtr, size_t count, int  dstDevice, cudaStream_t stream = 0 )
    Prefetches memory to the specified destination device.
__host__ ​cudaError_t cudaMemRangeGetAttribute ( void* data, size_t dataSize, cudaMemRangeAttribute attribute, const void* devPtr, size_t count )
    Query an attribute of a given memory range.
__host__ ​cudaError_t cudaMemRangeGetAttributes ( void** data, size_t* dataSizes, cudaMemRangeAttribute ** attributes, size_t numAttributes, const void* devPtr, size_t count )
    Query attributes of a given memory range.

__host__ ​cudaError_t cudaMemcpy2D ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )
    Copies data between host and device.
__host__ ​cudaError_t cudaMemcpy2DArrayToArray ( cudaArray_t dst, size_t wOffsetDst, size_t hOffsetDst, cudaArray_const_t src, size_t wOffsetSrc, size_t hOffsetSrc, size_t width, size_t height, cudaMemcpyKind kind = cudaMemcpyDeviceToDevice )
    Copies data between host and device.
__host__ ​ __device__ ​cudaError_t cudaMemcpy2DAsync ( void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )
    Copies data between host and device.
__host__ ​cudaError_t cudaMemcpy2DFromArray ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind )
    Copies data between host and device.
__host__ ​cudaError_t cudaMemcpy2DFromArrayAsync ( void* dst, size_t dpitch, cudaArray_const_t src, size_t wOffset, size_t hOffset, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )
    Copies data between host and device.
__host__ ​cudaError_t cudaMemcpy2DToArray ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind )
    Copies data between host and device.
__host__ ​cudaError_t cudaMemcpy2DToArrayAsync ( cudaArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, cudaMemcpyKind kind, cudaStream_t stream = 0 )
    Copies data between host and device.
__host__ ​cudaError_t cudaMemcpy3D ( const cudaMemcpy3DParms* p )
    Copies data between 3D objects.
__host__ ​ __device__ ​cudaError_t cudaMemcpy3DAsync ( const cudaMemcpy3DParms* p, cudaStream_t stream = 0 )
    Copies data between 3D objects.
__host__ ​cudaError_t cudaMemcpy3DPeer ( const cudaMemcpy3DPeerParms* p )
    Copies memory between devices.
__host__ ​cudaError_t cudaMemcpy3DPeerAsync ( const cudaMemcpy3DPeerParms* p, cudaStream_t stream = 0 )
    Copies memory between devices asynchronously.
__host__ ​ __device__ ​cudaError_t cudaMemcpyAsync ( void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream = 0 )
    Copies data between host and device.
__host__ ​cudaError_t cudaMemcpyFromSymbol ( void* dst, const void* symbol, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyDeviceToHost )
    Copies data from the given symbol on the device.
__host__ ​cudaError_t cudaMemcpyFromSymbolAsync ( void* dst, const void* symbol, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0 )
    Copies data from the given symbol on the device.
__host__ ​cudaError_t cudaMemcpyPeer ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count )
    Copies memory between two devices.
__host__ ​cudaError_t cudaMemcpyPeerAsync ( void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count, cudaStream_t stream = 0 )
    Copies memory between two devices asynchronously.
__host__ ​cudaError_t cudaMemcpyToSymbol ( const void* symbol, const void* src, size_t count, size_t offset = 0, cudaMemcpyKind kind = cudaMemcpyHostToDevice )
    Copies data to the given symbol on the device.
__host__ ​cudaError_t cudaMemcpyToSymbolAsync ( const void* symbol, const void* src, size_t count, size_t offset, cudaMemcpyKind kind, cudaStream_t stream = 0 )
    Copies data to the given symbol on the device.
__host__ ​cudaError_t cudaMemset ( void* devPtr, int  value, size_t count )
    Initializes or sets device memory to a value.
__host__ ​cudaError_t cudaMemset2D ( void* devPtr, size_t pitch, int  value, size_t width, size_t height )
    Initializes or sets device memory to a value.
__host__ ​ __device__ ​cudaError_t cudaMemset2DAsync ( void* devPtr, size_t pitch, int  value, size_t width, size_t height, cudaStream_t stream = 0 )
    Initializes or sets device memory to a value.
__host__ ​cudaError_t cudaMemset3D ( cudaPitchedPtr pitchedDevPtr, int  value, cudaExtent extent )
    Initializes or sets device memory to a value.
__host__ ​ __device__ ​cudaError_t cudaMemset3DAsync ( cudaPitchedPtr pitchedDevPtr, int  value, cudaExtent extent, cudaStream_t stream = 0 )
    Initializes or sets device memory to a value.
__host__ ​ __device__ ​cudaError_t cudaMemsetAsync ( void* devPtr, int  value, size_t count, cudaStream_t stream = 0 )
    Initializes or sets device memory to a value.
__host__ ​cudaExtent make_cudaExtent ( size_t w, size_t h, size_t d )
    Returns a cudaExtent based on input parameters.
__host__ ​cudaPitchedPtr make_cudaPitchedPtr ( void* d, size_t p, size_t xsz, size_t ysz )
    Returns a cudaPitchedPtr based on input parameters.
__host__ ​cudaPos make_cudaPos ( size_t x, size_t y, size_t z )
    Returns a cudaPos based on input parameters.
*/

#endif
