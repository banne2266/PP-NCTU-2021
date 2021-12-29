#define BLOCK_SIZE 16

__kernel void convolution(const int filterWidth, const int imageHeight, const int imageWidth,
    __global const float *inputImage, __constant float *filter,  __global float *outputImage) 
{
    int halffilterSize = filterWidth >> 1;
    int i = get_global_id(0);
    int j = get_global_id(1);
    int idX = get_local_id(0);
    int idY = get_local_id(1);
    int ii = i - halffilterSize;
    int jj = j - halffilterSize;
    
    int x = ii;
    int y = jj;

    __local float P[BLOCK_SIZE+7][BLOCK_SIZE+7];

    P[idX][idY] = inputImage[x * imageWidth + y];
    if(idX < filterWidth)
    {
        x = ii + BLOCK_SIZE;
        y = jj;
        P[idX + BLOCK_SIZE][idY] = inputImage[x * imageWidth + y];
    }
    if(idY < filterWidth)
    {
        x = ii;
        y = jj + BLOCK_SIZE;
        P[idX][idY + BLOCK_SIZE] = inputImage[x * imageWidth + y];
    }
    if(idX < filterWidth && idY < filterWidth){
        x = ii + BLOCK_SIZE;
        y = jj + BLOCK_SIZE;
        P[idX + BLOCK_SIZE][idY + BLOCK_SIZE] = inputImage[x * imageWidth + y];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float sum = 0.0f;
	for (x = 0; x < filterWidth; x++)
	{
		for (y = 0; y < filterWidth; y++)
		{
            sum += P[x+idX][y+idY] * filter[x*filterWidth + y];
		}
	}
    outputImage[i * imageWidth + j] = sum;
}
