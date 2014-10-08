
#include <stdio.h>

void resize( char* pIn, char* pOut, int widthIn, int heightIn, int widthOut, int heightOut );
void print( char* p, int width, int height );

void main()
{
	char img_in[4][4] = {{0,0,1,1}, {0,1,0,1}, {1,0,1,0}, {1,1,0,0}};

	char img_out[8][8] = {{0}};

	resize( &img_in[0][0], &img_out[0][0], 4, 4, 8, 8 );

	print( &img_out[0][0], 8, 8 );
}

void resize( char* pIn, char* pOut, int widthIn, int heightIn, int widthOut, int heightOut )
{
	for (int i=0;i<heightOut;i++)
	{
		int iIn = i * heightIn / heightOut;
		for (int j=0;j<widthOut;j++)
		{
			int jIn = j * widthIn / widthOut;
			pOut[ i*widthOut + j ] = pIn[ iIn*widthIn + jIn ];
		}
	}
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