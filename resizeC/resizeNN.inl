
template <typename T>
void resize( T* pIn, T* pOut, int widthIn, int heightIn, int widthOut, int heightOut )
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
