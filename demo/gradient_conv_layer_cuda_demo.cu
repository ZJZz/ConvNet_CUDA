#include <iostream>
#include <cstdio>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


__global__ void grad_conv_cu(int n_filter, int istart, int depth_offset, int n_depths, int ilayer, double *d_convgrad_work, double *d_this_delta, double *d_act_pre)
{
    int i_filt = blockIdx.x * blockDim.x + threadIdx.x; // ordinal number of the filter weight 0~8 ?
    // printf("i_filt: %d\n", i_filt);
    // if(i_filt > n_filter) return;
    if(i_filt >= n_filter) return;

    /*--------------input layer info--------------*/
    int in_height, in_width, in_channels;
    // if(ilayer == 0)
    // {   
    //     in_height   = d_img_height;
    //     in_width    = d_img_width;
    //     in_channels = d_img_channel; 
    // }
    // else
    // {
    //     in_height   = d_height[ilayer-1];
    //     in_width    = d_width[ilayer-1];
    //     in_channels = d_channel[ilayer-1];
    // }

    // Test
    in_height   = 5;
    in_width    = 5;
    in_channels = 1;

    /*--------------input layer info--------------*/


    // int ihid_offset = blockIdx.y; // 0~8 ?
    // int prod = d_width[ilayer] * d_height[ilayer];
    // int actual_start_slice = ihid_offset % n_depths + depth_offset;
    // int ihid_actual = ihid_offset / n_depths * d_depth[ilayer] + actual_start_slice; // the ordinal postion in the entire layer

    int ihid_offset = blockIdx.y; // 0~8 ?

    // printf("ihid_offset: %d\n", ihid_offset);
    int prod = 3 * 3;
    int actual_start_slice = ihid_offset % n_depths + depth_offset;
    int ihid_actual = ihid_offset / n_depths * 1 + actual_start_slice; // the ordinal postion in the entire layer
    // printf("ihid_actual: %d\n", ihid_actual);

    double delta;

    // if(i_filt == n_filter) // bias term
    // {
    //     delta = d_this_delta[blockIdx.z * d_nhid[ilayer] + ihid_actual];
    //     grad_ptr = d_convgrad_work + blockIdx.z * d_max_convgrad_each;
    //     grad_ptr[ihid_offset * d_convgrad_cols[ilayer] + d_n_prior_weights[ilayer]-1] = delta;
    //     return;
    // }

    // location of this kernel within the filter.
    // prod = (2 * d_HalfWidH[ilayer] + 1) * in_channels;
    // i_filt_row = i_filt / prod;
    // int k = i_filt - i_filt_row * prod;
    // i_filt_col = k / in_channels;
    // in_slice   = k % in_channels;

    // Test
    prod = (2 * 1 + 1) * in_channels;
    int i_filt_row = i_filt / prod;
    int k = i_filt - i_filt_row * prod;
    int i_filt_col = k / in_channels;
    int in_slice   = k % in_channels;

    // location of this neuron within the volume of the current layer
    // prod = d_width[ilayer] * d_depth[ilayer];
    // cur_row = ihid_actual / prod;
    // k = ihid_actual - cur_row * prod;
    // cur_col = k / d_depth[ilayer];

    prod = 3 * 1;
    int cur_row = ihid_actual / prod;
    k = ihid_actual - cur_row * prod;
    int cur_col = k / 1;

    // prior layer neuron
    // int in_row = d_strideV[ilayer] * cur_row - d_padV[ilayer] + i_filt_row;

    // Test
    int in_row = 2 * cur_row - 1 + i_filt_row;

    if(in_row < 0 || in_row >= in_height) return;

    // int in_col = d_strideH[ilayer] * cur_col - d_padH[ilayer] + i_filt_col;

    // Test
    int in_col = 2 * cur_col - 1 + i_filt_col;

    if(in_col < 0 || in_col >= in_width) return;

    // grad_ptr = d_convgrad_work + blockIdx.z * d_max_convgrad_each;

    // Test
    double *grad_ptr = d_convgrad_work;

        
    // delta = d_this_delta[blockIdx.z * d_nhid[ilayer] + ihid_actual];
    
    // Test
    delta = d_this_delta[blockIdx.z * 9 + ihid_actual];

    // input corresponding to this filter weight
    int i_in = (in_row * in_width + in_col) * in_channels + in_slice;

    // if(ilayer)
    //     input = d_act[ilayer-1][blockIdx.z * d_nhid[ilayer -1 ] + i_in];
    // else
    //     input = d_predictors[(istart + blockIdx.z) * d_n_pred + i_in];

    // Test

    double input = d_act_pre[blockIdx.z * 5 + i_in];
   

    
    // grad_ptr[ihid_actual * d_n_prior_weights[ilayer] + i_filt] = input * delta;
    
    // printf("%d\n", ihid_actual * 9 + i_filt);
    grad_ptr[ihid_actual * 9 + i_filt] = input * delta;

}

__global__ void device_flatten_gradient(int i_slcie_start, int max_depth, int ilayer, double *d_grad, double *d_convgrad_work)
{
    int i_prior = blockIdx.x * blockDim.x + threadIdx.x;
    // if(i_prior >= d_n_prior_weights[ilayer]) return;

    // Test
    if(i_prior >= 9) return;

    int i_slice = blockIdx.y;
    int i_case  = blockIdx.z;

    // float *grad_ptr = d_grad[ilayer] + i_case * d_n_weights;
    // float *work_ptr = d_convgrad_work + i_case * d_max_convgrad_each;

    // Test
    double *grad_ptr = d_grad;
    double *work_ptr = d_convgrad_work;

    double sum = 0.0;

    for(int i_row = 0; i_row < 3; i_row++)
    {
        for(int i_col = 0; i_col < 3; i_col++)
        {
            // neuron at i_row, i_col, i_slice
            // int k = (i_row * d_width[ilayer] + i_col) * max_depth + i_slice;
            // sum += work_ptr[k * d_convgrad_cols[ilayer] + i_prior];

            // Test
            int k = (i_row * 3 + i_col) * max_depth + i_slice;
            sum += work_ptr[k * 9 + i_prior];
            
        }
    }
    // printf("%lf\n", sum);

    // grad_ptr[(i_slice + i_slcie_start) * d_n_prior_weights[ilayer] + i_prior] = sum;
    grad_ptr[(i_slice + i_slcie_start) * 9 + i_prior] = sum;
}




/*

0 2 0 0 2            k          output         delta

1 2 2 0 1          0 0 -1       1  -4  -3     1  2  3

1 0 0 0 2         -1 -1 1      -5  -1  -2     4  5  6

2 0 1 2 0          0 -1 0       0  -3  -4     7  8  9

2 2 1 2 2


result

298 206 344 
496 562 146 
112 297 12 

*/

int main()
{

    int thread_per_block = 9;
    dim3 block_launch;
    block_launch.x = 1;
    block_launch.y = 9;
    block_launch.z = 1;

    double h_activation_pre[25] = {0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 1.0, 2.0, 2.0};
    double h_grad_work[81] = {0.0};
    double h_this_delta[9] = {1, 32, 27, 100, 5, 24, 0, 72, 144};

    double *d_activation_pre, *d_convgrad_work, *d_this_delta;

    cudaMalloc((double **)&d_activation_pre, sizeof(double) * 25);
    cudaMalloc((double **)&d_convgrad_work, sizeof(double) * 81);
    cudaMalloc((double **)&d_this_delta, sizeof(double) * 9);

    cudaMemcpy(d_activation_pre, h_activation_pre,  sizeof(double) * 25, cudaMemcpyHostToDevice);
    cudaMemcpy(d_convgrad_work, h_grad_work, sizeof(double) * 81, cudaMemcpyHostToDevice);
    cudaMemcpy(d_this_delta, h_this_delta, sizeof(double) * 9, cudaMemcpyHostToDevice);


    grad_conv_cu<<<block_launch, thread_per_block>>>(9, 3, 0, 1, 4, d_convgrad_work, d_this_delta, d_activation_pre);

    cudaMemcpy(h_grad_work, d_convgrad_work, sizeof(double) * 81, cudaMemcpyDeviceToHost);

    // for(int i = 0; i < 81; i++)
    // {
    //     std::cout << h_grad_work[i] << " ";
    //     if((i + 1) % 9 == 0)
    //         std::cout << std::endl;
    // }


    cudaDeviceSynchronize();

    /*---------------------*/

    double h_grad[9] = {0.0};
    
    double *d_grad;


    cudaMalloc((double **)&d_grad, sizeof(double) * 9);
    cudaMemcpy(d_grad, h_grad, sizeof(double) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_convgrad_work, h_grad_work, sizeof(double) * 81, cudaMemcpyHostToDevice);

    int thread_per_block_flatten = 9;
    dim3 block_launch_flatten; 
    block_launch_flatten.x = 9;
    block_launch_flatten.y = 1;
    block_launch_flatten.z = 1;

    device_flatten_gradient<<< block_launch_flatten, thread_per_block_flatten>>>(0, 1, 3, d_grad, d_convgrad_work);

    // gpuErrchk( cudaPeekAtLastError() );

    cudaMemcpy(h_grad, d_grad, sizeof(double) * 9, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 9; i++)
    {
        std::cout << h_grad[i] << " ";
        if((i + 1) % 3 == 0)
            std::cout << std::endl;
    }


    // for(int i = 0; i < 9; i++)
    // {
    //     double grad = 0;
    //     for(int j = 0; j < 9; j++)
    //     {
    //         grad += h_grad_work[j * 9 + i];
    //     }
    //     std::cout << grad << " ";
    //     if((i + 1) % 3 == 0)
    //         std::cout << std::endl;
    // }

    return 0;
}