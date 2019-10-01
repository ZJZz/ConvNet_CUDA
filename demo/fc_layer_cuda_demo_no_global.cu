#include <cstdio>

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
