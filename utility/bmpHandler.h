#ifndef BMPHANDLER
#define BMPHANDLER

typedef unsigned char byte;

#pragma pack(2)

/*WORD:two-byte type*/
typedef unsigned short WORD;
/*DWORD:four-byte type*/
typedef unsigned long DWORD;

/*file head of bitmap*/
typedef struct BMP_FILE_HEADER
{
	WORD bType;             /*  file identifier          */
	DWORD bSize;            /*  file size                */
	WORD bReserved1;        /*  retention value,must 0   */       
	WORD bReserved2;        /*  retention value,must 0   */
	DWORD bOffset;          /*  The offset from the last file header to the start of image data bits. */
	BMP_FILE_HEADER(int width=1, int height=1)
	{
		bType = 19778;// 0x4D42
		bOffset = sizeof(struct BMP_FILE_HEADER) + 40;//sizeof(BMPINF);

		bReserved1 = bReserved2 = 0;

		bSize = width * height * 3 + bOffset;
	}
} BMPFILEHEADER;

/*bitmap header*/
typedef struct BMP_INFO
{
	DWORD bInfoSize;       /*  size of the message header */
	DWORD bWidth;          /*  width of the image         */
	DWORD bHeight;         /*  height of the image        */
	WORD bPlanes;          /*  number of bit-plane image  */
	WORD bBitCount;        /*  number of bits per pixel   */
	DWORD bCompression;    /*  compression Type           */
	DWORD bmpImageSize;    /*  image size, in bytes       */
	DWORD bXPelsPerMeter;  /*  horizontal resolution      */
	DWORD bYPelsPerMeter;  /*  vertical resolution        */
	DWORD bClrUsed;        /*  number of colors used      */
	DWORD bClrImportant;   /*  significant number of colors*/

	BMP_INFO(int width=1, int height=1)
	{
		bPlanes		= 1;
		bBitCount	= 24;
		bXPelsPerMeter	= 11811;
		bYPelsPerMeter	= 11811;
		bInfoSize	= sizeof(BMP_INFO);

		bCompression = bClrUsed = bClrImportant = 0;

		bWidth	= width;
		bHeight	= height;
		bmpImageSize = width * height * 3;
	}
} BMPINF;

#pragma pack()

class BMPHandler
{
public:
	BMPHandler();

	template <typename T>
	static void saveImage( char* filename, 
		T* B_Out, T* G_Out, T* R_Out, 
		int height, int width );

	static void saveImage( char* filename, unsigned int *buffer, 
		int height, int width );

	// 读取一幅图像的像素值
	template <typename T>
	static void readImageData(char* filename, 
		T *B_P, T *G_P, T *R_P);

	static void readImageData(char* filename, unsigned int *buffer);

	// 读取一幅图像的文件头
	static void getImageSize(char* filename, int& height, int& width );


	static void bgr2Int( unsigned int* BGR, 
		byte* B, byte* G, byte* R,
		int width, int height,
		bool b2Int = true );	// if false, reverse

};


template <typename T>
void BMPHandler::readImageData(char* filename, T *B_P, T *G_P, T *R_P) 
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
	bmpImageSize = infoHeader.bmpImageSize;
	width = infoHeader.bWidth;
	height = infoHeader.bHeight;
	bitCount = infoHeader.bBitCount;
	bytesPerPixel = infoHeader.bBitCount / 8;

	byte *Pixel;

	fseek(fp,offset,SEEK_SET);

	//Assign addresses
	Pixel =(byte *)malloc((size-offset)*sizeof(byte));

	i=0;

	byte* B=(byte *)malloc(height*width*sizeof(float));
	byte* G=(byte *)malloc(height*width*sizeof(float));
	byte* R=(byte *)malloc(height*width*sizeof(float));

	while(1)
	{
		*Pixel=fgetc(fp);
		B[i]= *Pixel;
		*Pixel=fgetc(fp);
		G[i]= *Pixel;
		*Pixel=fgetc(fp);
		R[i]= *Pixel;

		i++;
		if(i==height*width)
		{
			break;
		}
	}

	t=0; 
	for(i=height-1;i>=0;i--)
	{
		for(j=0;j<width;j++)
		{
			B_P[i*width+j]=*(B+t*width+j);
			G_P[i*width+j]=*(G+t*width+j);
			R_P[i*width+j]=*(R+t*width+j);
		}
		t=t+1;
	}

	free(Pixel); Pixel = NULL;
	free( B ); free( G ); free( R );
}

template <typename T>
void BMPHandler::saveImage( char* filename, 
	T* B_Out, T* G_Out, T* R_Out, int height, int width )
{
	
	T *B_P,*G_P,*R_P;	//	缓存
	B_P=(T *)malloc(height*width*sizeof(T));
	G_P=(T *)malloc(height*width*sizeof(T));
	R_P=(T *)malloc(height*width*sizeof(T));

	FILE* fp=fopen( filename,"wb");
	if(fp == NULL)
	{
		printf("Cann't open the file in save!\n");

		free(B_P);
		free(G_P);
		free(R_P);

		return;
	}

	BMPFILEHEADER fileHeader(width,height);
	BMPINF infoHeader(width,height);

	fseek(fp, 0, 0);
	fwrite(&fileHeader, sizeof(fileHeader), 1, fp);
	fwrite(&infoHeader, sizeof(infoHeader), 1, fp);
	fseek(fp,fileHeader.bOffset,SEEK_SET);

	for(int i=height-1, t=0;i>=0;i--, t++)
	{
		for(int j=0;j<width;j++)
		{
			B_P[i*width+j] = B_Out[t*width+j];
			G_P[i*width+j] = G_Out[t*width+j];
			R_P[i*width+j] = R_Out[t*width+j];
		}
	}

	for (int i = 0;i<width*height; i++ )
	{
		fputc( B_P[i], fp );
		fputc( G_P[i], fp );
		fputc( R_P[i], fp );
	}

	fclose(fp);

	free(B_P);
	free(G_P);
	free(R_P);
}


#endif