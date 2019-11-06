

#define CUDA_KERNEL_LOOP(i, n) \
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)



// CUDA: use 512 threads per block
 const int CUDA_NUM_THREADS = 512;

 // CUDA: number of blocks for threads.
 inline int GET_BLOCKS(const int N)
{
   return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}





