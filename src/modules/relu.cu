#include "relu.h"
#include "device_util.h"
#include <cstdio>



__global__ void reluActivationForward(float* Z, float* A,
									  int Z_n_dim, int Z_c_dim, int Z_h_dim,  int Z_w_dim)
{
	//printf("In relu forward cuda kernel \n");
	CUDA_KERNEL_LOOP(index,  Z_n_dim * Z_c_dim * Z_h_dim * Z_w_dim)
	{

	    A[index] = Z[index] > 0 ? Z[index] : 0;
	    //printf("A:%f  Z:%f\n", A[index], Z[index]);
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

ReLU::ReLU(std::string name)
{
	std::cout << "Ctor ReLU" << std::endl;
	name_ = name;
}

ReLU::~ReLU()
{}


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
	std::cout << "In " << name_ << " backward" << std::endl;

	grad_output_ = grad_pre;

	if (grad_input_ == nullptr)
	{
		grad_input_ = new Tensor(input_->shape());
	}

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

