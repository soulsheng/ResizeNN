
#include <stdio.h>
#include <stdlib.h>
#include "resizeNN.cuh"
#include "bmpHandler.h"

#define	IMAGE_FILE_TEST		"DarkChannel.bmp"//720p - 1280��720
#define SCALE				2.0f//0.5f

void main()
{
	//char img_in[4][4] = {{0,1,2,2}, {3,0,1,2}, {4,3,0,1}, {4,4,3,0}};

	//char img_out[8][8] = {{0}};

	int width, height;
	BMPHandler::getImageSize( IMAGE_FILE_TEST, height, width );

	unsigned int *img_in=(unsigned int *)malloc(height*width*sizeof(unsigned int));
	unsigned int *img_out=(unsigned int *)malloc(height*SCALE*width*SCALE*sizeof(unsigned int));

	BMPHandler::readImageData( IMAGE_FILE_TEST, img_in );

	printf("\n 1.scale large: \n");
	CUResizeNN	large;
	large.initialize( width, height, width*SCALE, height*SCALE );
	large.process( img_in, img_out );
	BMPHandler::saveImage("outL.bmp", img_out, height*SCALE, width*SCALE );

	printf("\n 2.scale small: \n");
	CUResizeNN	small;
	small.initialize( width, height, width/SCALE, height/SCALE );
	small.process( img_in, img_out );
	BMPHandler::saveImage("outS.bmp", img_out, height/SCALE, width/SCALE );

	printf("\n 3.scale recover: \n");
	CUResizeNN	recover;
	recover.initialize( width/SCALE, height/SCALE, width, height );
	recover.process( img_out, img_in );
	BMPHandler::saveImage("outR.bmp", img_in, height, width );

	free( img_in );
	free( img_out );
}
