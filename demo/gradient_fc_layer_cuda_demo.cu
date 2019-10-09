#include <iostream>
#include <cstdio>


__global__ void grad_fc_layer_cu(int istart, int ilayer, int nc, double *pre_act, double *d_grad, double *d_this_delta)
{
    double *grad_ptr = NULL;
    
    int cur_neuron = blockIdx.x * blockDim.x + threadIdx.x; // 0,1,2

    double input;
    int cur_neruon_cnt;
    // if(ilayer == 0)
    //     cur_neruon_cnt = d_n_pred;
    // else
    //     cur_neruon_cnt = d_nhid[ilayer-1];

    // Test
    cur_neruon_cnt = 3;
    

    // input = pre_act
    // if(cur_neuron > cur_neruon_cnt) 
    //     return;
    // else if(cur_neuron == cur_neruon_cnt)
    //     input = 1.0;
    // else if(ilayer)
    //     input = d_act[ilayer-1][blockIdx.z * cur_neruon_cnt + cur_neuron];
    // else
    //     input = d_predictors[(istart + blockIdx.z) * cur_neruon_cnt + cur_neuron];

    input =  pre_act[blockIdx.z * cur_neruon_cnt + cur_neuron];

    int ihid  = blockIdx.y; // 0,1,2 current neuron cnt: 3
    // int ninp1 = cur_neruon_cnt + 1;
    
    // grad_ptr = d_grad[ilayer] + blockIdx.z * d_n_weights;

    // Test
    grad_ptr = d_grad;

    // grad_ptr[ihid*ninp1 + cur_neuron] = d_this_delta[blockIdx.z * d_nhid[ilayer] + ihid] * input;

    // Test
    grad_ptr[ihid * cur_neruon_cnt + cur_neuron] = d_this_delta[blockIdx.z * 3 + ihid] * input;
}


int main()
{
    int thread_per_block = 3;
    dim3 block_launch;
    block_launch.x = 1;
    block_launch.y = 3;
    block_launch.z = 1;

    double h_pre_act[3] = {1.0, 2.0, 3.0};
    double h_grad[9];
    double h_this_delta[3] = {28.0, 40.0, 54.0};

    double *d_pre_act, *d_grad, *d_this_delta;

    cudaMalloc((double **)&d_pre_act, sizeof(double) * 3);
    cudaMalloc((double **)&d_grad, sizeof(double) * 9);
    cudaMalloc((double **)&d_this_delta, sizeof(double) * 3);

    cudaMemcpy(d_pre_act,  h_pre_act,  sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad, h_grad, sizeof(double) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_this_delta, h_this_delta, sizeof(double) * 3, cudaMemcpyHostToDevice);


    grad_fc_layer_cu<<<block_launch, thread_per_block>>>(5, 5, 6, d_pre_act, d_grad, d_this_delta);

    cudaMemcpy(h_grad, d_grad, sizeof(double) * 9, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 9; i++)
    {
        std::cout << h_grad[i] << " ";
        if((i + 1) % 3 == 0)
            std::cout << std::endl;
    }
    return 0;
}