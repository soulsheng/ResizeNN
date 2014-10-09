
#include "bmpHandler.h"

#include <stdio.h>
#include <math.h>
#include <iostream>
using namespace std;

	BMPHandler::BMPHandler()
	{
	}


	void BMPHandler::getImageSize( char* filename, int& height, int& width )
	{
		FILE *fp=NULL;

		fp=fopen(filename,"rb");
		if(fp == NULL)
		{
			printf("Cann't open the file!\n");
			exit(0);
		}

		BMPFILEHEADER fileHeader;
		BMPINF infoHeader;

		fread(&fileHeader, sizeof(fileHeader), 1, fp);
		fread(&infoHeader, sizeof(infoHeader), 1, fp);

		width = infoHeader.bWidth;
		height = infoHeader.bHeight;

		fclose(fp);
	}

	void BMPHandler::readImageData( char* filename, unsigned int *buffer )
	{
		FILE *fp=NULL;
		FILE *fp_b=NULL;
		FILE *fp_br=NULL;
		BMPFILEHEADER fileHeader;
		BMPINF infoHeader;
		long offset, bmpImageSize, width, height, bytesPerPixel, size, bitCount;
		int  i,j;
		long t;

		//char *bmpArray;
		//unsigned char **p;
		//WORD c;

		fp=fopen( filename,"rb");
		if(fp == NULL)
		{
			printf("Cann't open the file!\n");
			exit(0);
		}

		fseek(fp, 0, 0);
		fread(&fileHeader, sizeof(fileHeader), 1, fp);
		fread(&infoHeader, sizeof(infoHeader), 1, fp);

		//Calculates and outputs the offset bitmap data, image size, width and height of bytes per pixel
		size = fileHeader.bSize;
		offset = fileHeader.bOffset;
		width = infoHeader.bWidth;
		height = infoHeader.bHeight;

		//byte *Pixel;
		//Pixel =(byte *)malloc((size-offset)*sizeof(byte));

		fseek(fp,offset,SEEK_SET);
		//fread( Pixel, sizeof(byte), size-offset, fp );

		int scanSize = 3 * width * sizeof(byte);
		byte *rdata = new byte[scanSize];

		for (int y = height - 1; y >= 0; y--)
		{
			fread(rdata, scanSize, 1, fp);
			// shuffle rgb
			int index = y * width;

			for (int i = 0; i < width; i++)
			{
				buffer[index + i] = (rdata[i * 3] << 16)
					| (rdata[i * 3 + 1] << 8) | rdata[i * 3 + 2] | 0xff000000;
			}
		}

		delete[] rdata;
	}

	void BMPHandler::saveImage( char* filename, unsigned int *buffer, int height, int width )
	{
		FILE* fp=fopen( filename,"wb");
		if(fp == NULL)
		{
			printf("Cann't open the file in save!\n");

			return;
		}

		BMPFILEHEADER fileHeader(width,height);
		BMPINF infoHeader(width,height);

		fseek(fp, 0, 0);
		fwrite(&fileHeader, sizeof(fileHeader), 1, fp);
		fwrite(&infoHeader, sizeof(infoHeader), 1, fp);
		fseek(fp,fileHeader.bOffset,SEEK_SET);

		int scanSize = 3 * width * sizeof(byte);
		byte *rdata = new byte[scanSize];

		for (int y = height - 1; y >= 0; y--)
		{
			int index = y * width;

			for (int i = 0; i < width; i++)
			{
				unsigned int val = buffer[y * width + i];

				rdata[i*3]		= ( val>> 16 ) & 0x000000ff;	// b
				rdata[i*3+1]	= ( val>> 8 ) & 0x000000ff;	// g
				rdata[i*3+2]	= ( val ) & 0x000000ff;		// r
			}
			fwrite(rdata, scanSize, 1, fp);
		}

		fclose(fp);

	}


	void BMPHandler::bgr2Int( unsigned int* RGBA, byte* B, byte* G, byte* R, int width, int height, bool b2Int /*= true */ )
	{
		for (int y = 0; y < height; y++)
		{
			for (int i = 0; i < width; i++)
			{
				int index = y * width + i;
				if ( b2Int )
				{
					RGBA[index] = (B[index] << 16)
						| (G[index] << 8) | R[index] | 0xff000000;
				}
				else
				{
					unsigned int val = RGBA[index];
					R[index] = ( val ) & 0x000000ff;
					G[index] = ( val>> 8 ) & 0x000000ff;
					B[index] = ( val>> 16 ) & 0x000000ff;
				}		
			}
		}
	}

