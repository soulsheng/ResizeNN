
#include "helper_timer.h"
#define	ENABLE_TIMER		1

template <typename T>
void resize( T* pIn, T* pOut, int widthIn, int heightIn, int widthOut, int heightOut )
{

#if ENABLE_TIMER
	StopWatchInterface *timer = 0;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
#endif

	for (int i=0;i<heightOut;i++)
	{
		int iIn = i * heightIn / heightOut;
		for (int j=0;j<widthOut;j++)
		{
			int jIn = j * widthIn / widthOut;
			pOut[ i*widthOut + j ] = pIn[ iIn*widthIn + jIn ];
		}
	}


#if ENABLE_TIMER
	sdkStopTimer(&timer);
	printf("Processing time ALL: %f (ms)\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);
#endif
}
