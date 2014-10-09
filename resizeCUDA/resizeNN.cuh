
#pragma once

//template< typename T >
class CUResizeNN
{
public:
	CUResizeNN();
	~CUResizeNN();

	void process( unsigned int* pIn, unsigned int* pOut, bool bDeviceBuffer = false );

	void initialize( int widthIn, int heightIn, int widthOut, int heightOut );

	void release();

	static void process( float* pIn, float* pOut, int widthIn, int heightIn, int widthOut, int heightOut );
	static void process( unsigned int* pIn, unsigned int* pOut, int widthIn, int heightIn, int widthOut, int heightOut );

private:
	int widthIn, heightIn, widthOut, heightOut;

	unsigned int *pIn_d, *pOut_d;

};