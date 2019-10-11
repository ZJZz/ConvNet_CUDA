#include <iostream>
#include <cstdio>
#include <cmath>

#define REDUC_THREADS 256
#define REDUC_BLOCKS  64

double output[3] = {0,0,0}; // one case in a batch

double labels[3] = {0.0 , 1.0, 0.0};

double loss;

__global__ void device_softmax(int istart, int istop, double *d_outptr)
{
    int icase, iout;
    double *outptr, sum;

    icase = blockIdx.x * blockDim.x + threadIdx.x; // threads bind with one case in a batch

    if(icase >= istop - istart) return;

    outptr = d_outptr + (icase + istart) * 3; // output vector for this case

    sum = 0.0;

    for(iout = 0; iout < 3; iout++)
    {
        if(outptr[iout] < 300.0)
            outptr[iout] = exp(outptr[iout]);
        else
            outptr[iout] = 300.0;
        
        sum += outptr[iout];
    }

    for(iout = 0; iout < 3; iout++)
        outptr[iout] /= sum;
}

int cuda_softmax(int istart, int istop)
{
    // int n = istop - istart;

    // int threads_per_block = (n + warpsize -1) / warpsize * warpsize;
    // if(threads_per_block > 4 * warpsize)
    //     threads_per_block = 4 * warpsize;

    // int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;



    double *d_outptr;

    cudaMalloc((double **)&d_outptr, sizeof(double) * 3);

    cudaMemcpy(d_outptr,  output,  sizeof(double) * 3, cudaMemcpyHostToDevice);

    device_softmax<<<1, 1>>>(istart, istop, d_outptr);

    cudaMemcpy(output, d_outptr, sizeof(double) * 3, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 3; i++)
        std::cout << output[i] << " ";
    std::cout << std::endl;
    // 0.333333 0.333333 0.333333

    cudaDeviceSynchronize();

    return 0;
}

__global__ void device_loss(double* d_ll_out, double *d_outptr)
{
    __shared__ double partial_ll;
    int i, n , n_classes, index;
    double sum_ll;

    index = threadIdx.x;

    n = 1;
    n_classes = 3;

    sum_ll = 0.0;

    for(i = blockIdx.x * blockDim.x + index; i < n; i += blockDim.x * gridDim.x)
    {
        // sum_ll -= log(d_outptr[i*n_classes + d_class[i]] + 1.e-30);
        sum_ll -= log(d_outptr[i*n_classes + 1] + 1.e-30); 
    }
    partial_ll = sum_ll;

    __syncthreads();

    // for(i = blockDim.x >> 1; i; i >>= 1)
    // {
    //     if(index < i)
    //     {
    //         partial_ll += partial_ll[index + i];
    //     }
    //     __syncthreads();
    // }

    if(index == 0)
        *d_ll_out = partial_ll;
} 

int cuda_loss(int n, double *ll)
{
    int i, blocks_per_grid;
    double sum;


    double reduc_fdata;

    blocks_per_grid = (n + REDUC_THREADS - 1) / REDUC_THREADS;

    if(blocks_per_grid > REDUC_BLOCKS)
        blocks_per_grid = REDUC_BLOCKS;

    double h_ll_out;

    

    double *d_outptr;
    double *d_ll_out;

    cudaMalloc((double **)&d_outptr, sizeof(double) * 3);
    cudaMalloc((double **)&d_ll_out, sizeof(double));

    cudaMemcpy(d_outptr,  output,  sizeof(double) * 3, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ll_out,  &h_ll_out,  sizeof(double), cudaMemcpyHostToDevice);

    device_loss<<< 1, 1>>>(d_ll_out, d_outptr);

    cudaDeviceSynchronize();

    cudaMemcpy(&reduc_fdata, d_ll_out, sizeof(double), cudaMemcpyDeviceToHost);

    sum = 0.0;

    // for(i = 0; i < blocks_per_grid; i++)
    //     sum += reduc_fdata;

    for(i = 0; i < 1; i++)
        sum += reduc_fdata;

    *ll = sum;

    return 0;

}


int main()
{
    cuda_softmax(0, 1);
    cuda_loss(1, &loss);
    std::cout << loss << std::endl; // // 1.09861
}