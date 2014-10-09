
#pragma once

//template< typename T >
class CUResizeNN
{
public:
	CUResizeNN( int widthIn, int heightIn, int widthOut, int heightOut );
	~CUResizeNN();

	void process( unsigned int* pIn, unsigned int* pOut );

	void initialize();

	void release();

private:
	int widthIn, heightIn, widthOut, heightOut;

	unsigned int *pIn_d, *pOut_d;

};