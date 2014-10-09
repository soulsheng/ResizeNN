
#include "resizeNN.cuh"
#include <cuda_runtime.h>

#include "helper_timer.h"
#define	ENABLE_TIMER		1

/*	size:	720p(1280*720)
	gpu:	gtx480
	cpu:	E8200

		scaleX/Y	time(ms)
		1->0.5		0.39		
		0.5->1		0.35		*/

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
void resizeGPU( T* pIn_d, T* pOut_d, int widthIn, int heightIn, int widthOut, int heightOut )
{
	dim3 block(16, 16);
	dim3 grid( (widthOut+15)/16, (heightOut+15)/16 );
	resize_kernel<<< grid, block >>>( pIn_d, pOut_d, widthIn, heightIn, widthOut, heightOut );
}

void CUResizeNN::process( unsigned int* pIn, unsigned int* pOut, bool bDeviceBuffer /*= false*/ )
{

#if ENABLE_TIMER
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
#endif
	
	if( !bDeviceBuffer )
		cudaMemcpy( pIn_d, pIn, widthIn*heightIn*sizeof(unsigned int), cudaMemcpyHostToDevice );
	else
		cudaMemcpy( pIn_d, pIn, widthIn*heightIn*sizeof(unsigned int), cudaMemcpyDeviceToDevice );

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time cudaMemcpyHostToDevice: %f (ms)\n", sdkGetTimerValue(&timer));
#endif

	
#if ENABLE_TIMER
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif

	resizeGPU<unsigned int>( pIn_d, pOut_d, widthIn, heightIn, widthOut, heightOut );
	
	cudaDeviceSynchronize();

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time kernel: %f (ms)\n", sdkGetTimerValue(&timer));
#endif

		
#if ENABLE_TIMER
	sdkResetTimer(&timer);
	sdkStartTimer(&timer);
#endif
	
	if( !bDeviceBuffer )
		cudaMemcpy( pOut, pOut_d, widthOut*heightOut*sizeof(unsigned int), cudaMemcpyDeviceToHost );
	else
		cudaMemcpy( pOut, pOut_d, widthOut*heightOut*sizeof(unsigned int), cudaMemcpyDeviceToDevice );

#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time cudaMemcpyDeviceToHost: %f (ms)\n", sdkGetTimerValue(&timer));
#endif

	
#if ENABLE_TIMER
	sdkDeleteTimer(&timer);
#endif

}

CUResizeNN::CUResizeNN( int widthIn, int heightIn, int widthOut, int heightOut )
{
	this->widthIn = widthIn;
	this->heightIn = heightIn;
	this->widthOut = widthOut;
	this->heightOut = heightOut;

	initialize();
}

CUResizeNN::~CUResizeNN()
{
	release();
}

void CUResizeNN::initialize()
{
	cudaMalloc( &pIn_d, widthIn*heightIn*sizeof(unsigned int) );
	cudaMalloc( &pOut_d, widthOut*heightOut*sizeof(unsigned int) );
}

void CUResizeNN::release()
{
	cudaFree( pIn_d );
	cudaFree( pOut_d );
}