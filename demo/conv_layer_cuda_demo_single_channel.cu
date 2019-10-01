#include <cstdio>

__global__ void Conv2d_cu(float *input, float *filter, float *output)
{

    /*-------------current layer info-------------------*/
    
    int cur_layer_width = 3;
    
    // current activation neuron
    // decode from block and thread index
    int out_slice   = blockIdx.x * blockDim.x + threadIdx.x;
    int out_row     = blockIdx.y / cur_layer_width;
    int out_col     = blockIdx.y % cur_layer_width;
    
    /*-------------current layer info-------------------*/


    /*-------------prior layer info---------------------*/
    int in_height  = 5;
    int in_width   = 5;
    int in_channel = 1;
    /*-------------prior layer info---------------------*/

    /*-------------conv info----------------------------*/
    // Filter Size
    int d_HalfWidH = 1;
    int d_HalfWidV = 1;
    int filter_size = 2 * d_HalfWidH + 1;

    float *wt_ptr = filter;

    // Stride 
    int d_strideV = 2;
    int d_strideH = 2;

    // Padding
    int d_padV    = 1;
    int d_padH    = 1;
    /*-------------conv info----------------------------*/

    
    /*--------------prior layer activation field----------*/
    int rbase, rstart, rstop, cbase, cstart, cstop;
    rbase = rstart = d_strideV * out_row - d_padV;
    rstop = rstart + 2 * d_HalfWidV;
    cbase = cstart = d_strideH * out_col - d_padH;
    cstop = cstart + 2 * d_HalfWidH;

    if (rstart < 0) rstart = 0;
    if (cstart < 0) cstart = 0;

    if (rstop >= in_height) rstop = in_height - 1;
    if (cstop >= in_width)  cstop = in_width  - 1;
    /*--------------prior layer location field----------*/


    /*--------------compute-----------------------------*/
    float *act_ptr = input;

    float sum = 0.0;

    for (int in_row = rstart; in_row <= rstop ; in_row++) 
    {
        int k_wt = (in_row - rbase) * filter_size ;
        int k_in = in_row * in_width ;
        
        for (int in_col = cstart ; in_col <= cstop ; in_col++) 
        {
            int wt_sub = (k_wt + in_col - cbase) * in_channel;
            
            int in_sub = (k_in + in_col) * in_channel;
            
            for (int in_slice = 0 ; in_slice < in_channel; in_slice++) 
            {
                // need relative position, so need rbase and cbase
                // wt_sub = ((in_row - rbase) * filter_size + in_col - cbase) * in_channel + in_slice ;
                // in_sub = (in_row * in_width + in_col ) * in_channel + in_slice ;
                printf("Input location:  (%d, %d) - > %.2f  *  Filter location: (%d, %d) - > %.2f \n", in_row, in_col, act_ptr[in_sub], wt_sub / 3, wt_sub % 3, wt_ptr[wt_sub]);
                sum += act_ptr[in_sub] * wt_ptr[wt_sub];
                ++wt_sub;
                ++in_sub;
            } // For in_slice
        } // For in_col
     } // For in_row


     // bias


     // activation

    /*--------------compute-----------------------------*/



    /*--------------save result-----------------------------*/
    // encode back to activation layout
    // int out_height = 3;
    int out_width  = 3;
    int out_channel  = 1;

    int ihid = (out_row * out_width + out_col) * out_channel + out_slice;   // Activity for any layer type is (height, width, depth)
    
    float *actptr =  output;
    actptr[ihid] = sum ;
    /*--------------save result-----------------------------*/


}


int main()
{
    float h_input[25] = {0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 1.0, 2.0, 2.0};
    float h_filter[9] = {0.0, 0.0, -1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 0.0};
    float h_res[9];

    float *d_input, *d_filter, *d_res;

    cudaMalloc((float **)&d_input, sizeof(float) * 25);
    cudaMalloc((float **)&d_filter, sizeof(float) * 9);
    cudaMalloc((float **)&d_res, sizeof(float) * 9);

    cudaMemcpy(d_input,  h_input,  sizeof(float) * 25, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(float) * 9, cudaMemcpyHostToDevice);


    dim3 block_launch;

    block_launch.x = 1;
    block_launch.y = 9;
    block_launch.z = 1; 

    Conv2d_cu<<< block_launch, 1 >>>(d_input, d_filter, d_res);

    cudaMemcpy(h_res, d_res, sizeof(float) * 9, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 9; i++)
    {
        printf("%.2f  ", h_res[i]);
        if((i + 1) % 3 == 0)
            printf("\n");
    }


    return 0;
}
