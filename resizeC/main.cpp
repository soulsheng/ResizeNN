
#include <stdio.h>
#include <stdlib.h>
#include "resizeNN.h"
#include "bmpHandler.h"

#define	IMAGE_FILE_TEST		"DarkChannel.bmp"
#define SCALE				0.5f//2.0f

void main()
{
	//char img_in[4][4] = {{0,1,2,2}, {3,0,1,2}, {4,3,0,1}, {4,4,3,0}};

	//char img_out[8][8] = {{0}};

	int width, height;
	BMPHandler::getImageSize( IMAGE_FILE_TEST, height, width );

	unsigned int *img_in=(unsigned int *)malloc(height*width*sizeof(unsigned int));
	unsigned int *img_out=(unsigned int *)malloc(height*SCALE*width*SCALE*sizeof(unsigned int));

	BMPHandler::readImageData( IMAGE_FILE_TEST, img_in );

	resize( img_in, img_out, width, height, width*SCALE, height*SCALE );

	BMPHandler::saveImage("outd5.bmp", img_out, height*SCALE, width*SCALE );

	free( img_in );
	free( img_out );
}
