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

	// ��ȡһ��ͼ����ļ�ͷ
	static void readImageHeader(char* filename, BMPFILEHEADER& fileHeader, BMPINF& infoHeader);

	// ��ȡһ��ͼ�������ֵ
	template <typename T>
	static void readImageData(char* filename, T *B, T *G, T *R, T **B_P, T **G_P, T **R_P);

	template <typename T>
	static void saveImage( char* filename, 
		T* B_Out, T* G_Out, T* R_Out, 
		int height, int width );

	template <typename T>
	static void saveImage( char* filename, 
		T** B_Out, T** G_Out, T** R_Out, 
		int height, int width );

	static void saveImage( char* filename, unsigned int *buffer, 
		int height, int width );

	// ��ȡһ��ͼ�������ֵ
	template <typename T>
	static void readImageData(char* filename, 
		T *B, T *G, T *R, T *B_P, T *G_P, T *R_P);

	template <typename T>
	static void readImageData(char* filename, 
		T *B_P, T *G_P, T *R_P);

	static void readImageData(char* filename, unsigned int *buffer);

	// ��ȡһ��ͼ����ļ�ͷ
	static void getImageSize(char* filename, int& height, int& width );


	static void bgr2Int( unsigned int* BGR, 
		byte* B, byte* G, byte* R,
		int width, int height,
		bool b2Int = true );	// if false, reverse

private:
	//static BMPFILEHEADER fileHeader;
	//static BMPINF infoHeader;
};

template <typename T>
void BMPHandler::readImageData(char* filename, T *B, T *G, T *R, T *B_P, T *G_P, T *R_P) 
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
	
	while(1)
	{
		//printf("%d ", *Pixel);
		//*Pixel=fgetc(fp);	

		*Pixel=fgetc(fp);
		B[i]=(T)*Pixel;
		*Pixel=fgetc(fp);
		G[i]=(T)*Pixel;
		*Pixel=fgetc(fp);
		R[i]=(T)*Pixel;
		
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

	free(Pixel);
}

template <typename T>
void BMPHandler::readImageData(char* filename, T *B, T *G, T *R, T **B_P, T **G_P, T **R_P) 
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

	while(1)
	{
		*Pixel=fgetc(fp);
		B[i]=(T)*Pixel;
		*Pixel=fgetc(fp);
		G[i]=(T)*Pixel;
		*Pixel=fgetc(fp);
		R[i]=(T)*Pixel;

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
			B_P[i][j]=*(B+t*width+j);
			G_P[i][j]=*(G+t*width+j);
			R_P[i][j]=*(R+t*width+j);
		}
		t=t+1;
	}

	free(Pixel); Pixel = NULL;
}


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
	
	T *B_P,*G_P,*R_P;	//	����
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

template <typename T>
void BMPHandler::saveImage( char* filename, 
	T** B_Out, T** G_Out, T** R_Out, 
	int height, int width )
{
	T **B_P,**G_P,**R_P;	//	����
	B_P=(T **)malloc(height*sizeof(T*));
	G_P=(T **)malloc(height*sizeof(T*));
	R_P=(T **)malloc(height*sizeof(T*));
	for (int i=0;i<height;i++)
	{
		B_P[i]=(T *)malloc(width*sizeof(T));
		G_P[i]=(T *)malloc(width*sizeof(T));
		R_P[i]=(T *)malloc(width*sizeof(T));
	}

	FILE* fp=fopen( filename,"wb");
	if(fp == NULL)
	{
		printf("Cann't open the file in save!\n");

		for (int i=0;i<height;i++)
		{
			free( B_P[i] );
			free( G_P[i] );
			free( R_P[i] );
		}
		free( B_P ); free( G_P ); free( R_P );

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
			B_P[i][j] = B_Out[t][j];
			G_P[i][j] = G_Out[t][j];
			R_P[i][j] = R_Out[t][j];
		}
	}

	for (int i = 0;i<height; i++ )
	{
		for (int j = 0;j<width; j++ )
		{
			fputc( B_P[i][j], fp );
			fputc( G_P[i][j], fp );
			fputc( R_P[i][j], fp );
		}
	}

	fclose(fp);

	for (int i=0;i<height;i++)
	{
		free( B_P[i] );
		free( G_P[i] );
		free( R_P[i] );
	}
	free( B_P ); free( G_P ); free( R_P );
}

//�ҶȻ�
void rgbGray(float **rgb_gray,float **R_P,float **G_P,float **B_P,int height,int width) ;
void rgbGray(int *rgb_gray,float *R_P,float *G_P,float *B_P,int height,int width) ;

template <typename T>
void rgbGray( int **rgb_gray,T **R_P,T **G_P,T **B_P,int height,int width )
{
	for(int i=0;i<height;i++)
	{
		for(int j=0;j<width;j++)
		{
			//Converting an RGB image into a grayscale image
			float gray = 0.299f * R_P[i][j] + 0.587f * G_P[i][j] + 0.114f * B_P[i][j]; 
			rgb_gray[i][j] = (int)( gray + 0.5f );
		}
	}    
}

template <typename T>
void rgbGray512( int *rgb_gray,T **R_P,T **G_P,T **B_P,int height,int width )
{
	for(int i=0;i<height;i++)
	{
		for (int j=0;j<width;j++)
		{
			T r = R_P[i][j];
			T g = G_P[i][j];
			T b = B_P[i][j];

			int	a1 = (int)floor(r/32.0);
			int	a2 = (int)floor(g/32.0);
			int	a3 = (int)floor(b/32.0);

			rgb_gray[i*width+j] = a1*64+a2*8+a3;
		}
	}
}

template <typename T>
void rgbGray512( int *rgb_gray,T *R_P,T *G_P,T *B_P,int height,int width )
{
	for(int i=0;i<height;i++)
	{
		for (int j=0;j<width;j++)
		{
			T r = R_P[i*width+j];
			T g = G_P[i*width+j];
			T b = B_P[i*width+j];

			int	a1 = (int)floor(r/32.0);
			int	a2 = (int)floor(g/32.0);
			int	a3 = (int)floor(b/32.0);

			rgb_gray[i*width+j] = a1*64+a2*8+a3;
		}
	}
}

#endif