
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

private:
	int widthIn, heightIn, widthOut, heightOut;

	unsigned int *pIn_d, *pOut_d;

};