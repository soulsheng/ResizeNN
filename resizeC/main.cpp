
#include <stdio.h>
#include "resizeNN.h"
#include "bmpHandler.h"

void print( char* p, int width, int height );

void main()
{
	char img_in[4][4] = {{0,1,2,2}, {3,0,1,2}, {4,3,0,1}, {4,4,3,0}};

	char img_out[8][8] = {{0}};

	resize( &img_in[0][0], &img_out[0][0], 4, 4, 8, 8 );

	print( &img_out[0][0], 8, 8 );
}

void print( char* p, int width, int height )
{
	for (int i=0;i<height;i++)
	{
		for (int j=0;j<width;j++)
		{
			printf("%5d,", p[i*width+j]);
		}
		printf("\n");
	}
	printf("\n");
}