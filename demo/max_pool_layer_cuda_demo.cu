#include <iostream>
#include <cstdio>
#include <limits>
#include <algorithm>
#include <cmath>


/*

poolsize: 2*2 strid: 2

1  1  2  4
5  6  7  8                    6  8
3  2  1  0    ---MaxPool--->  3  4
1  2  3  4

*/

__global__ void MaxPool_cu(float* input, float* output, int *max_id)
{
    
    /*-------------current layer info-------------------*/
    int out_width   = 2;
    // int out_height  = 2;
    int out_channel = 1;

    int out_row = blockIdx.y / out_width;
    int out_col = blockIdx.y % out_width;
    int out_slice = blockIdx.x * blockDim.x + threadIdx.x;

    int out_idx = (out_row * out_width + out_col) * out_channel + out_slice;
    /*-------------current layer info-------------------*/

    
    /*-------------pool info----------------------------*/
    int d_poolWidH = 2;
    int d_poolWidV = 2;
    int d_strideH  = 2;
    int d_strideV  = 2;
    /*-------------pool info----------------------------*/


    /*--------------prior layer activation field----------*/
    int rstart = d_strideV * out_row;
    int rstop  = rstart + d_poolWidV - 1;
    int cstart = d_strideH * out_col;
    int cstop  = cstart + d_poolWidH - 1;
    /*--------------prior layer activation field----------*/


    /*-------------prior layer info-------------------*/
    float *act_ptr = input;
    int in_width   = 4;
    int in_channel = 1;
    /*-------------prior layer info-------------------*/


    /*-------------compute-------------------*/
    int *poolmax_id = &max_id[out_idx];
    float value = -1.e60;

    for(int in_row = rstart; in_row <= rstop; in_row++)
    {
        for(int in_col = cstart; in_col <= cstop; in_col++)
        {
            float x = act_ptr[(in_row * in_width + in_col) * in_channel + out_slice];
            if(x > value)
            {
                value = x;
                *poolmax_id = in_row * in_width + in_col;
            }
        }
    }
    /*-------------compute-------------------*/


    /*--------------save result--------------*/
    act_ptr = output;
    act_ptr[out_idx] = value;
    /*--------------save result--------------*/
}


int main()
{

    float h_input[16] = {1.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0};
    float h_out[4];
    int h_poolmax_id[4];


    float *d_input, *d_out;
    int *d_poolmax_id;
    cudaMalloc((float **)&d_input, sizeof(float) * 16);
    cudaMalloc((float **)&d_out, sizeof(float) * 4);
    cudaMalloc((int **)&d_poolmax_id, sizeof(int) * 4);

    cudaMemcpy(d_input,  h_input,  sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out,    h_out, sizeof(float) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_poolmax_id,    h_out, sizeof(int) * 4, cudaMemcpyHostToDevice);

    dim3 block_launch;

    block_launch.x = 1;
    block_launch.y = 4;
    block_launch.z = 1; 

    MaxPool_cu<<< block_launch, 1 >>>(d_input, d_out, d_poolmax_id);

    cudaMemcpy(h_out, d_out, sizeof(float) * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_poolmax_id, d_poolmax_id, sizeof(int) * 4, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 4; i++)
    {
        printf("%.2f  ", h_out[i]);
        if((i + 1) % 2 == 0)
            printf("\n");
    }

    for(int i = 0; i < 4; i++)
    {
        printf("%d  ", h_poolmax_id[i]);
        if((i + 1) % 2 == 0)
            printf("\n");
    }

    return 0;
}