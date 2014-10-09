
#include "resizeNN.cuh"
#include <cuda_runtime.h>

#include "helper_timer.h"
#define	ENABLE_TIMER		1

template <typename T>
__global__ void resize_kernel( T* pIn, T* pOut, int widthIn, int heightIn, int widthOut, int heightOut )
{
	int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;

	if( i < heightOut && j < widthOut )
	{
		int iIn = i * heightIn / heightOut;
		int jIn = j * widthIn / widthOut;
		pOut[ i*widthOut + j ] = pIn[ iIn*widthIn + jIn ];
	}
}

template <typename T>
void resizeGPU( T* pIn, T* pOut, int widthIn, int heightIn, int widthOut, int heightOut )
{
	T *pIn_d, *pOut_d;
	cudaMalloc( &pIn_d, widthIn*heightIn*sizeof(T) );

#if ENABLE_TIMER
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
#endif

	cudaMalloc( &pIn_d, widthIn*heightIn*sizeof(T) );
	cudaMalloc( &pOut_d, widthOut*heightOut*sizeof(T) );

	cudaMemcpy( pIn_d, pIn, widthIn*heightIn*sizeof(T), cudaMemcpyHostToDevice );

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time cudaMemcpyHostToDevice: %f (ms)\n", sdkGetTimerValue(&timer));
#endif

#if ENABLE_TIMER
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

	dim3 block(16, 16);
	dim3 grid( (widthOut+15)/16, (heightOut+15)/16 );
	resize_kernel<<< block, grid >>>( pIn_d, pOut_d, widthIn, heightIn, widthOut, heightOut );

	cudaDeviceSynchronize();

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time kernel: %f (ms)\n", sdkGetTimerValue(&timer));
#endif

#if ENABLE_TIMER
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

	cudaMemcpy( pOut, pOut_d, widthOut*heightOut*sizeof(T), cudaMemcpyDeviceToHost );
	
#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time cudaMemcpyDeviceToHost: %f (ms)\n", sdkGetTimerValue(&timer));
#endif

	cudaFree( pIn_d );
	cudaFree( pOut_d );

#if ENABLE_TIMER
	sdkDeleteTimer(&timer);
#endif
}

void resize( unsigned int* pIn, unsigned int* pOut, int widthIn, int heightIn, int widthOut, int heightOut )
{
	resizeGPU<unsigned int>( pIn, pOut, widthIn, heightIn, widthOut, heightOut );
}