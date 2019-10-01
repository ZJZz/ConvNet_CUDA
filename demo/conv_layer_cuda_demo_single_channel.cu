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


// __global__ void device_hidden_activation_LOCAL_CONV (
//     int local_vs_conv , // Is this a LOCAL (vs CONV) layer?
//     int case_start ,    // First case in this batch (relative to dataset)
//     int case_offset ,   // Offset relative to this batch (used in shared version)
//     int slice_start ,   // First slice in this batch
//     int n_slices ,      // Number of slices to be done in this launch
//     int ilayer          // Layer to process
//     )
//  {
//     int kwt, kin, wtsub, insub, iheight, iwidth, idepth, n_height, n_width, n_depth, wt_cols, ihid ;
//     int rstart, rstop, cstart, cstop, rbase, cbase, in_slice, in_row, in_col, nH ;
//     float *f_inptr, *wptr ;
//     double sum, *actptr ;
 
//     // decode thread info back to location
//     idepth = blockIdx.x * blockDim.x + threadIdx.x ;
 
//     if (idepth >= n_slices)
//        return ;
 
//     idepth += slice_start ;
//     iheight = blockIdx.y / d_width[ilayer] ;
//     iwidth = blockIdx.y % d_width[ilayer] ;
 
//     nH = 2 * d_HalfWidH[ilayer] + 1 ;
 
//     // We are about to compute the activation of neuron (iheight, iwidth, idepth) in this layer.
//     // Note that it is critical that idepth be associated with the thread.
//     // This ensures that adjacent threads reference the same input, which allows efficient memory use.
//     // Also, the weights are ordered so that depth-fastest changes produce perfect or very good coalescing.
//     // Thus, neuron layout in current layer is (height, width, depth).
//     // This gives strong motivation for LOCAL layers to have depth a multiple of 32.
//     // To see why, note the ihid= below.  That multiplication ensures perfect coalescing of the weight fetches.
 
//     // icase = blockIdx.z ;   // Avoid having to declare this (and use a register) by directly referencing it later
 
//     wt_cols = d_depth_cols[ilayer] ; // Padded size of weight matrix rows; each has depth[ilayer] data values, then zero padding
//                                     // There are n_prior_weights rows (prior depth * (2*HalfWidH+1) * (2*HalfWidV+1) + 1)
//                                     // A convolutional layer has a different weight set for each slice,
//                                     // but the same weight set for all neurons (visual field placement) in a slice.
//     wptr = d_weights[ilayer] + idepth ; // First filter weight for this slice is here; subsequent weights spaced by wt_cols
       
 
//     sum = 0.0 ;
 
//     // Center of first filter is at HalfWidth-Pad; filter begins at -Pad.
//     rbase = rstart = d_strideV[ilayer] * iheight - d_padV[ilayer] ;
//     rstop = rstart + 2 * d_HalfWidV[ilayer] ;
//     cbase = cstart = d_strideH[ilayer] * iwidth - d_padH[ilayer] ;
//     cstop = cstart + 2 * d_HalfWidH[ilayer] ;
 
//     if (rstart < 0)
//        rstart = 0 ;
//     if (cstart < 0)
//        cstart = 0 ;
 
//     // It's annoying and messy, but we must duplicate the same code for the case of this being the
//     // first hidden layer (fed by the input) versus a subsequent hidden layer (fed by prior activations).
//     // This is because the input uses a float pointer, and activations a double pointer.
//     // Deciding in the inner loop would be too slow!
   
//     actptr = d_act[ilayer-1] + (blockIdx.z + case_offset) * d_nhid[ilayer-1] ;
//     n_height = d_height[ilayer-1] ;
//     n_width = d_width[ilayer-1] ;
//     n_depth = d_depth[ilayer-1] ;
//     if (rstop >= n_height)
//         rstop = n_height - 1 ;
//     if (cstop >= n_width)
//         cstop = n_width - 1 ;

//     for (in_row=rstart ; in_row<=rstop ; in_row++) {
//         kwt = (in_row - rbase) * nH ;
//         kin = in_row*n_width ;
//         for (in_col=cstart ; in_col<=cstop ; in_col++) {
//             wtsub = (kwt + in_col - cbase) * n_depth ;
//             insub = (kin+in_col) * n_depth ;
//             for (in_slice=0 ; in_slice<d_depth[ilayer-1] ; in_slice++) {
//             // wtsub = ((in_row - rbase) * nH + in_col - cbase) * n_depth + in_slice ;
//             // insub = (in_row*n_width+in_col)*n_depth+in_slice ;
//             sum += actptr[insub] * wptr[wtsub*wt_cols] ;
//             ++wtsub ;
//             ++insub ;
//             } // For in_slice
//             } // For in_col
//         } // For in_row

//     sum += wptr[(d_n_prior_weights[ilayer]-1) * wt_cols] ;      // Bias
       
    
//     // activation function
//     if (sum > MAX_EXP)
//        sum = 1.0 ;
//     else {
//        sum = exp ( 2.0 * sum ) ;
//        sum = (sum - 1.0) / (sum + 1.0) ;
//        }
 
//     // encode back to activation layout
//     n_height = d_height[ilayer] ;
//     n_width = d_width[ilayer] ;
//     n_depth = d_depth[ilayer] ;
//     actptr = d_act[ilayer] ;
//     ihid = (iheight * n_width + iwidth) * n_depth + idepth ;   // Activity for any layer type is (height, width, depth)
    
//     actptr[(blockIdx.z+case_offset)*d_nhid[ilayer]+ihid] = sum ;
//  }