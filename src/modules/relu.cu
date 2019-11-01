#include "relu.h"
#include <cstdio>

 #define CUDA_KERNEL_LOOP(i, n) \
   for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
        i < (n); \
        i += blockDim.x * gridDim.x)

__global__ void reluActivationForward(float* Z, float* A,
									  int Z_n_dim, int Z_c_dim, int Z_h_dim,  int Z_w_dim)
{
	//printf("In relu forward cuda kernel \n");
	CUDA_KERNEL_LOOP(index,  Z_n_dim * Z_c_dim * Z_h_dim * Z_w_dim)
	{

	    A[index] = Z[index] > 0 ? Z[index] : 0;
	    printf("A:%f  Z:%f\n", A[index], Z[index]);
	}
}

__global__ void reluActivationBackward(float* Z, float* dA, float* dZ,
		                               int Z_n_dim, int Z_c_dim, int Z_h_dim,  int Z_w_dim)
{

	CUDA_KERNEL_LOOP(index, Z_n_dim * Z_c_dim * Z_h_dim * Z_w_dim)
	{
	    dZ[index] = dA[index] * (Z[index] > 0);
	}
}

// CUDA: use 512 threads per block
 const int CUDA_NUM_THREADS = 512;

 // CUDA: number of blocks for threads.
 int GET_BLOCKS(const int N)
 {
   return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
 }

ReLU::ReLU(std::string name)
{
	std::cout << "Ctor ReLU" << std::endl;
	name_ = name;
}

ReLU::~ReLU()
{
	if(input_ != nullptr)
		delete input_;
	if(output_ != nullptr)
		delete output_;
	if(grad_input_ != nullptr)
		delete grad_input_;
	if(grad_output_ != nullptr)
		delete grad_output_;
}


Tensor* ReLU::forward(Tensor* input)
{
	std::cout << "In ReLU Host forward" << std::endl;


	input_ = input;
	//std::cout << "-----" << std::endl;

	if (output_ == nullptr)
	{
		output_ = new Tensor(input_->shape());
	    output_->allocateMemoryIfNotAllocated(input->shape());
	}
	else
	{
		std::cout << "output_ already allocated" << std::endl;
	}

	int count = input_->len();

	//std::cout << "count: " << count << std::endl;


	//std::cout << input_ << std::endl;

	//input->print_tensor("input", true, 1, 3);

	// CUDA
	reluActivationForward<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
														 input_->get_device_ptr().get(),
														 output_->get_device_ptr().get(),
														 input_->n(),
														 input_->c(),
														 input_->h(),
														 input_->w());

//	std::cout << "Exit ReLU Host forward" << std::endl;
//
//	output_->print_tensor("output_", true, 1, 3);

	return output_;

}

Tensor* ReLU::backward(Tensor* grad_pre)
{

	grad_input_->allocateMemoryIfNotAllocated(input_->shape());

	int count = input_->len();

	// CUDA
	reluActivationBackward<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(input_->get_device_ptr().get(),
															grad_output_->get_device_ptr().get(),
															grad_input_->get_device_ptr().get(),
															input_->n(),
															input_->c(),
															input_->h(),
															input_->w());


	return 	grad_input_;

}

