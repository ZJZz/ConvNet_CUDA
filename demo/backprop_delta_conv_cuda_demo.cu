#include <cstdio>
#include <iostream>


__global__ void backprop_delta_from_conv_cu(int ilayer, double *this_d_ptr, double *prior_d_ptr, double *w_ptr)
{
    /* Backprop Delta Equation
        delta_i_cur = f'(act_i_cur) * sigma( w_k_i_next * delta_k_next )
    */
    int cur_layer_neuron = blockIdx.x * blockDim.x + threadIdx.x;


    // if(cur_layer_neuron >= d_nhid[ilayer]) return;

    // Test
    if(cur_layer_neuron >= 25) return;

    /*------------compute coordinates in current layer------------*/
    // int prod      = d_width[ilayer] * d_channel[ilayer];
    // int cur_row   = cur_layer_neuron / prod;
    // int k         = cur_layer_neuron - cur_row * prod;
    // int cur_col   = k / d_channel[ilayer];
    // int cur_slice = k % d_channel[ilayer];
    // int cur_case  = blockIdx.y;
    
    // Test
    int prod      = 5 * 1;
    int cur_row   = cur_layer_neuron / prod;
    int k         = cur_layer_neuron - cur_row * prod;
    int cur_col   = k / 1;
    int cur_slice = k % 1;
    int cur_case  = 1;

    
    
    /*------------compute coordinates in current layer------------*/

    
    // double *this_delta_ptr  = d_this_delta + icase * d_nhid[ilayer+1]; // (next layer, known)
    // double *prior_delta_ptr = d_prior_delta + icase * d_nhid[ilayer];  // (curent layer, unknown)
    double *this_delta_ptr   = this_d_ptr;
    double *prior_delta_ptr  = prior_d_ptr;

    /*-------------conv info-----------------------*/
    // int kernel_size_H = 2 * d_HalfWidH[ilayer+1] + 1;   // Horizontal filter size
    // int strideV       = d_strideV[ilayer+1];
    // int strideH       = d_strideH[ilayer+1];
    // int padV          = d_padV[ilayer+1];
    // int padH          = d_padH[ilayer+1];

    // Test
    int kernel_size_H = 2 * 1 + 1;   // Horizontal filter size
    int strideV       = 2;
    int strideH       = 2;
    int padV          = 1;
    int padH          = 1;
    /*-------------conv info-----------------------*/

    /*-------------next layer info------------*/
    // int next_type    = d_layer_type[ilayer+1];
    // int next_height  = d_height[ilayer+1]; 
    // int next_width   = d_width[ilayer+1];
    // int next_channel = d_depth[ilayer+1];

    // Test
    int next_height  = 3; 
    int next_width   = 3;
    int next_channel = 1;
    /*-------------next layer info------------*/

    /*
    Unlike we do reverse direction in serial code, here, we computre the delta just follow the equation, from ilayer to next layer.
    Becasue in parallel computing, every neuron's delta in current layer compute indenpendently, so we can complete computation the delta in one thread.
    The critical point here is to figure out every neuron in current layer contributions to which neuron in next layer.
    To compute those neuron in the next layer, we need to compute the bounday in the next layer. 
    */

    // review the common find prior rectangle equation
    // this >= next * stride - pad(start)           => next <= (this + pad) / stride
    // this <= next * stride - pad(start) + 2 * hw  => next >= (this + pad - 2 * hw) / stride

    // compute next row info
    // int next_row_stop  = cur_row + padV;
    // int k = next_row_stop - 2 * d_HalfWidV[ilayer+1];
    // int next_row_start = k;
    // next_row_stop  /= strideV;
    // next_row_start /= strideV;

    // Test
    int next_row_stop  = cur_row + padV;
    k = next_row_stop - 2 * 1;
    int next_row_start = k;
    next_row_stop  /= strideV;
    next_row_start /= strideV;

    if(k >= 0 && k % strideV) // if the division about was inexact
        ++next_row_start;     // we must move past fractional part
    
    // stay inside the layer boundary
    if(next_row_stop >= next_height)
        next_row_stop = next_height - 1;
    if(next_row_start < 0)
        next_row_start = 0;

    // compute next col info
    // int next_col_stop = cur_col + padH;
    // k = next_col_start = next_col_stop - 2 * d_HalfWidH[ilayer+1];
    // next_col_stop  /= strideH;
    // next_col_start /= strideH;

    int next_col_stop = cur_col + padH;
    int next_col_start = next_col_stop - 2 * 1;
    k = next_col_start;
    next_col_stop  /= strideH;
    next_col_start /= strideH;
    
    if(k >= 0 && k % strideH)
        ++next_col_start;
    
    if(next_col_stop >= next_width)
        next_col_stop = next_width - 1;
    
    if(next_col_stop  < 0)
        next_col_start = 0;

    // Get weight connect current layer to next layer
    // weights = d_weights[ilayer+1];
    double *weights = w_ptr;

    // length of these padded weight vectors
    // wt_cols = d_depth_cols[ilayer+1];

    double sum = 0.0;

    for(int next_row = next_row_start; next_row <= next_row_stop; next_row++)
    {
        for(int next_col = next_col_start; next_col <= next_col_stop; next_col++)
        {
            // starting coordinates of rectangle in current layer
            // to compute the postion of the current neuron in the filter
            int cur_row_start = strideV * next_row - padV;
            int cur_col_start = strideH * next_col - padH;
            // since we know the limit of next layer, we don't need the cur_row_stop, cur_col_stop
            
            for(int next_slice = 0; next_slice < next_channel; next_slice++)
            {
                // the ordinal postion of the current neuron in the next layer
                int idx_next = (next_row * next_width + next_col) * next_channel + next_slice;

                // filter weight
                double* w_ptr = weights + next_slice;

                // the ordinal postion of the current neuron in the filter
                // int idx_filter = ((cur_row - cur_row_start) * kernel_size_H + cur_col - cur_col_start) * d_depth[ilayer] + cur_slice;
                int idx_filter = ((cur_row - cur_row_start) * kernel_size_H + cur_col - cur_col_start) * 1 + cur_slice;

                printf("idx_filter: %d\n",idx_filter);

                // printf("this_d: %d * w_ptr: %d\n",this_delta_ptr[idx_next], w_ptr[idx_filter]);
                
                sum += this_delta_ptr[idx_next] * w_ptr[idx_filter];
            }
        }
    }

    
    

    // if(d_layer_type[ilayer] == TYPE_FC ||  d_layer_type[ilayer] == TYPE_CONV)
    // {
    //     cur_act = d_act[ilayer][icase * d_nhid[ilayer] + ihid];
    //     sum *= 1.0 - cur_act * cur_act;
    // }
    
    prior_delta_ptr[cur_layer_neuron] = sum;
    // printf("cur_row: %d cur_col: %d cur_slice: %d, prior_delta_ptr[%d]: %lf\n",cur_row, cur_col, cur_slice, cur_layer_neuron, sum);
    /*---*/
    /*---*/
    /*---*/
}


/*

stride = 2
pad    = 1

(ilayer)                     (ilayer + 1) this_delta

0 0 0 0 0 0 0                  

0 x x x x x 0                 1 2 3

0 x x x x x 0      kernel     4 5 6

0 x x x x x 0      1 2 3      7 8 9

0 x x x x x 0      4 5 6

0 x x x x x 0      7 8 9

0 0 0 0 0 0 0


result

5  14  10  24  15 
16 40  26  60  36 
20 44  25  54  30 
46 100 56  120 66 
35 74  40  84  45
*/

int main()
{
    int thread_per_block = 25;
    dim3 block_launch;
    block_launch.x = 1;
    block_launch.y = 1;
    block_launch.z = 1;

    double h_prior_d[25];
    double h_weight_ptr[9] = {1,2,3,4,5,6,7,8,9};
    double h_this_d[9] = {1,2,3,4,5,6,7,8,9};

    double *d_prior_d, *d_weight_ptr, *d_this_d;

    cudaMalloc((double **)&d_prior_d, sizeof(double) * 25);
    cudaMalloc((double **)&d_weight_ptr, sizeof(double) * 9);
    cudaMalloc((double **)&d_this_d, sizeof(double) * 9);

    cudaMemcpy(d_prior_d,  h_prior_d,  sizeof(double) * 25, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_ptr, h_weight_ptr, sizeof(double) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_this_d, h_this_d, sizeof(double) * 9, cudaMemcpyHostToDevice);


    backprop_delta_from_conv_cu<<<block_launch, thread_per_block>>>(5, d_this_d, d_prior_d, d_weight_ptr);

    cudaMemcpy(h_prior_d, d_prior_d, sizeof(double) * 25, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 25; i++)
    {
        std::cout << h_prior_d[i] << " ";
        if((i + 1) % 5 == 0)
            std::cout << std::endl;
    }

}