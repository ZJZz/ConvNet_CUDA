#include <cstdio>


// __constant__ float *d_weights; 


__global__ void Linear_cu(float *weights, float *input, float *res)
{
    int n_inputs, ihid;
    float sum;
    float *wt_ptr;
    float *in_ptr;

    ihid = threadIdx.x;

    

    n_inputs = 3;

    in_ptr = input; 
    wt_ptr = weights + ihid * n_inputs;


    sum = 0.0;
    for(int i_input = 0; i_input < n_inputs; i_input++)
    {
        printf("In thread %d\n", ihid);
        printf("%f * %f\n", *wt_ptr, in_ptr[i_input]);
        sum += *wt_ptr++ * in_ptr[i_input];
    }

    // bias
    // sum += *wt_ptr;

    res[ihid] = sum;

}

int main()
{
    float h_input[3] = {1.0, 2.0, 3.0}; 
    float h_weight[6] = {4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    float h_res[2];

    float *d_input, *d_weight, *d_res;

    cudaMalloc((float **)&d_input, sizeof(float) * 3);
    cudaMalloc((float **)&d_weight, sizeof(float) * 6);
    cudaMalloc((float **)&d_res, sizeof(float) * 2);

    cudaMemcpy(d_input,  h_input,  sizeof(float) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight, sizeof(float) * 6, cudaMemcpyHostToDevice);

    Linear_cu<<< 1, 2 >>>(d_weight, d_input, d_res);

    cudaMemcpy(h_res, d_res, sizeof(float) * 2, cudaMemcpyDeviceToHost);

    printf("%f %f\n",h_res[0], h_res[1]);


    return 0;
}

// __global__ void device_hidden_activation_FC (
//     int istart ,       // First case in this batch
//     int istop ,        // One past last case
//     int ilayer         // Layer to process
//     )
//  {
//     int icase, ihid, i_input, n_inputs, nhid_cols ;
//     float *f_inptr, *wptr ;
//     double sum, *actptr, *d_inptr ;
 
//     ihid = blockIdx.x * blockDim.x + threadIdx.x ;
 
//     if (ihid >= d_nhid[ilayer])
//        return ;
 
//     nhid_cols = d_nhid_cols[ilayer] ;
 
//     icase = blockIdx.y ;
 
//     wptr = d_weights[ilayer] + ihid ;  // Device weights are transpose of host weights, with this neuron changing fastest
//     sum = 0.0 ;
 
//     n_inputs = d_nhid[ilayer-1] ;
//     d_inptr = d_act[ilayer-1] + icase*n_inputs ;
//     for (i_input=0 ; i_input<n_inputs ; i_input++) 
//     {
//         sum += *wptr * d_inptr[i_input] ;
//         wptr += nhid_cols ;
//     }
//     sum += *wptr ;   // Bias
    
//     actptr = d_act[ilayer] ;
//     actptr[icase*d_nhid[ilayer]+ihid] = sum ;
//  }
 
//  int cuda_hidden_activation_FC (
//     int istart ,    // First case in this batch
//     int istop ,     // One past last case
//     int nhid ,      // Number of hidden neurons in this layer
//     int ilayer      // Layer to process
//     )
//  {
//     int warpsize, threads_per_block ;
//     char msg[256] ;
//     dim3 block_launch ;
//     cudaError_t error_id ;
 
//     warpsize = deviceProp.warpSize ;      // Threads per warp, likely 32 well into the future
 
//     threads_per_block = (nhid + warpsize - 1) / warpsize * warpsize ;
//     if (threads_per_block > 4 * warpsize)
//        threads_per_block = 4 * warpsize ;
 
//     block_launch.x = (nhid + threads_per_block - 1) / threads_per_block ;
//     block_launch.y = istop - istart ;
//     block_launch.z = 1 ;
 
//     device_hidden_activation_FC <<< block_launch , threads_per_block >>> ( istart , istop , ilayer ) ;   
 
//     // This does not trigger an escape, but it keeps the message queue running
//     user_pressed_escape () ;
 
//     cudaDeviceSynchronize() ;
//     error_id = cudaGetLastError () ;
//     if (error_id != cudaSuccess) {
//        sprintf_s ( msg , 255 , "cuda_hidden_activation_FC launch error %d: %s", error_id, cudaGetErrorString(error_id) ) ;
//        audit ( msg ) ;
//        MEMTEXT ( msg ) ;
//        return 1 ;
//        }
 
//     return 0 ;
//  }