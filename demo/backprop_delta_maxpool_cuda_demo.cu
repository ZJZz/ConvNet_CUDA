#include <cstdio>
#include <iostream>


/*
poolsize: 2*2 strid: 2

1  1  2  4
5  6  7  8                    6  8
3  2  1  0    ---MaxPool--->  3  4
1  2  3  4

poolmax-id: 5 7 8 11
*/

__global__ void backprop_delta_from_conv(int ilayer, double *this_d_ptr, double *prior_d_ptr, int *poolmax_id)
{
    /* Backprop Delta Equation
        delta_i_cur = f'(act_i_cur) * sigma( w_k_i_next * delta_k_next )
    */
    int cur_layer_neuron = blockIdx.x * blockDim.x + threadIdx.x;

    // if(cur_layer_neuron >= d_nhid[ilayer]) return;

    /*------------compute coordinates in current layer------------*/
    // int prod      = d_width[ilayer] * d_channel[ilayer];
    // int cur_row   = cur_layer_neuron/ prod;
    // int k         = cur_layer_neuron - cur_row * prod;
    // int cur_col   = k / d_channel[ilayer];
    // int cur_slice = k % d_channel[ilayer];
    // int cur_case = blockIdx.y;
    
    // Test
    int prod      = 4 * 1;
    int cur_row   = cur_layer_neuron / prod;
    int k         = cur_layer_neuron - cur_row * prod;
    int cur_col   = k / 1;
    int cur_slice = k % 1;
    int cur_case  = blockIdx.y;
    /*------------compute coordinates in current layer------------*/

    
    // double *this_delta_ptr  = d_this_delta + icase * d_nhid[ilayer+1]; // (next layer, known)
    // double *prior_delta_ptr = d_prior_delta + icase * d_nhid[ilayer];  // (curent layer, unknown)

    // Test
    double *this_delta_ptr  = this_d_ptr;
    double *prior_delta_ptr = prior_d_ptr;

    /*-------------conv info-----------------------*/
    // int pool_size_H   = d_PoolWidH[ilayer+1];   // Horizontal filter size
    // int strideV       = d_strideV[ilayer+1];
    // int strideH       = d_strideH[ilayer+1];

    // Test
    int pool_size_H   = 2;   // Horizontal filter size
    int strideV       = 2;
    int strideH       = 2;
    /*-------------conv info-----------------------*/

    /*-------------next layer info------------*/
    // int next_type    = d_layer_type[ilayer+1];
    // int next_height  = d_height[ilayer+1]; 
    // int next_width   = d_width[ilayer+1];
    // int next_channel = d_depth[ilayer+1];

    // Test
    int next_height  = 2; 
    int next_width   = 2;
    int next_channel = 1;
    /*-------------next layer info------------*/

    /*
    Unlike we do reverse direction in serial code, here, we computre the delta just follow the equation, from ilayer to next layer.
    Becasue in parallel computing, every neuron's delta in current layer compute indenpendently, so we can complete computation the delta in one thread.
    The critical point here is to figure out every neuron in current layer contributions to which neuron in next layer.
    To compute those neuron in the next layer, we need to compute the bounday in the next layer. 
    */

    // review the common find prior rectangle equation
    // this >= next * stride           => next <= this / stride
    // this <= next * stride + poolwidth - 1  => next >= (this - poolwidth + 1) / stride

    int next_row_stop  = cur_row;
    k = next_row_stop - pool_size_H + 1;
    int next_row_start = k;
    next_row_stop  /= strideV;
    next_row_start /= strideV;

    if(k >= 0 && k % strideV)
        ++next_row_start;
    
    if(next_row_stop >= next_height)
        next_row_stop = next_height - 1;
    if(next_row_start < 0)
        next_row_start = 0;

    int next_col_stop = cur_col;
    k = next_col_stop - pool_size_H + 1;
    int next_col_start = k;
    next_col_stop  /= strideH;
    next_col_start /= strideH;
    
    if(k >= 0 && k % strideH)
        ++next_col_start;
    
    if(next_col_stop >= next_width)
        next_col_stop = next_width - 1;
    
    if(next_col_stop  < 0)
        next_col_start = 0;

    // Get weight connect current layer to next layer
    // weights = d_weights[ilayer+1]; no weight here

    // length of these padded weight vectors
    // wt_cols = d_depth_cols[ilayer+1];

    double sum = 0.0;

    // poolmax_id_ptr = d_poolmax_id[ilayer+1] + icase * d_nhid[ilayer+1];
    // cur_width      = d_width[ilayer]; 

    // Test
    int *poolmax_id_ptr = poolmax_id;
    int cur_width       = 4; 

    for(int next_row = next_row_start; next_row <= next_row_stop; next_row++)
    {
        for(int next_col = next_col_start; next_col <= next_col_stop; next_col++)
        {
            // // starting coordinates of rectangle in current layer
            // // to compute the postion of the current neuron in the filter
            // cur_row_start = strideV * next_row - padV;
            // cur_col_start = strideH * next_col - padH;
            // // since we know the limit of next layer, we don't need the cur_row_stop, cur_col_stop
            
            // for(int next_slice = 0; next_slice < next_channel; next_slice++)
            // {
            //     // the ordinal postion of the current neuron in the next layer
            //     int idx_next = (next_row * next_width + next_col) * next_channel + next_slice;

            //     // filter weight
            //     double w_ptr = weights + next_slice;

            //     // the ordinal postion of the current neuron in the filter
            //     int idx_filter = ((cur_row - cur_row_start) * kernel_size_H + cur_col - cur_col_start) * d_depth[ilayer] + cur_slice;
                
            //     sum += this_delta_ptr[idx_next] * w_ptr[idx_filter];
            // }
            int idx_next = (next_row * next_width + next_col) * next_channel + cur_slice;
            if(cur_row == poolmax_id_ptr[idx_next] / cur_width && cur_col == poolmax_id_ptr[idx_next] % cur_width)
                sum += this_delta_ptr[idx_next];
        }
    }
    

    // if(d_layer_type[ilayer] == TYPE_FC ||  d_layer_type[ilayer] == TYPE_CONV)
    // {
    //     cur_act = d_act[ilayer][icase * d_nhid[ilayer] + ihid];
    //     sum *= 1.0 - cur_act * cur_act;
    // }
    
    prior_delta_ptr[cur_layer_neuron] = sum;
    /*---*/
    /*---*/
    /*---*/
}

/*
poolsize: 2*2 strid: 2

1  1  2  4
5  6  7  8                    6  8
3  2  1  0    ---MaxPool--->  3  4
1  2  3  4

poolmax-id: 5 7 8 11
*/


int main()
{

    int thread_per_block = 16;
    dim3 block_launch;
    block_launch.x = 1;
    block_launch.y = 1;
    block_launch.z = 1;

    double h_prior_d[16];
    double h_this_d[4] = {1,1,1,1};
    int    poolmax_id[4] = {5, 7, 8, 11};

    double *d_prior_d, *d_this_d;
    int    *d_poolmax_id;

    cudaMalloc((double **)&d_prior_d, sizeof(double) * 16);
    cudaMalloc((double **)&d_this_d,  sizeof(double) * 4);
    cudaMalloc((int **)&d_poolmax_id, sizeof(int) * 4);

    cudaMemcpy(d_prior_d,  h_prior_d,  sizeof(double) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_this_d, h_this_d, sizeof(double) * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_poolmax_id, poolmax_id, sizeof(int) * 4, cudaMemcpyHostToDevice);

    backprop_delta_from_conv<<< block_launch, thread_per_block>>>(5, d_this_d, d_prior_d, d_poolmax_id);

    cudaMemcpy(h_prior_d, d_prior_d, sizeof(double) * 16, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 16; i++)
    {
        std::cout << h_prior_d[i] << " ";
        if((i+1) % 4 == 0)
            std::cout << std::endl;
    }

    return 0;
}