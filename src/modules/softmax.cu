#include "softmax.h"
#include "device_util.h"
#include <cfloat>
#include <algorithm>



__global__ void kernel_channel_max(int num, int channels, int spatial_dim,
		                           float* data, float* out)
{
	CUDA_KERNEL_LOOP(index , num * spatial_dim)
	{
		int n = index / spatial_dim;
		int s = index % spatial_dim;
		float maxval = -FLT_MAX;

		for(int c = 0; c < channels; ++c)
		{
			maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
		}

		out[index] = maxval;
	}
}

__global__ void kernel_channel_subtract(int count, int num, int channels, int spatial_dim,
		                                float* channel_max, float* data)
{
	CUDA_KERNEL_LOOP(index , count)
	{
		int n = index / channels / spatial_dim;
		int s = index % spatial_dim;
		data[index] -= channel_max[n * spatial_dim + s];
	}
}


__global__ void kernel_exp(int count, float* data, float* out)
{
	CUDA_KERNEL_LOOP(index , count)
	{
		out[index] = exp(data[index]);
	}
}

__global__ void kernel_channel_sum(int num, int channels, int spatial_dim,
		                           float* data, float* channel_sum)
{
	CUDA_KERNEL_LOOP(index , num * spatial_dim)
	{
		int n = index / spatial_dim;
		int s = index % spatial_dim;

		float sum = 0;
		for(int c = 0; c < channels; ++c)
		{
			sum += data[(n * channels + c) * spatial_dim + s];
		}

		channel_sum[index] = sum;
	}
}

__global__ void kernel_channel_div(int count, int num, int channels, int spatial_dim,
		                           float* channel_sum, float* data)
{
	CUDA_KERNEL_LOOP(index , count)
	{
		int n = index / channels / spatial_dim;
		int s = index % spatial_dim;
		data[index] /= channel_sum[n * spatial_dim + s];
	}
}

__global__ void kernel_saxpy(int n, float alpha, float* x, float* y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = alpha * x[i] + y[i];
}

__global__ void kernel_sscale(int n, float scale, float * x)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		x[i] = scale * x[i];
	}
}

Softmax::Softmax(std::string name)
{
	name_ = name;
}

Softmax::~Softmax() { }

Tensor* Softmax::forward(Tensor* input)
{
	if (input_ == nullptr || batch_size_ != input->n())
	{
		input_ = input;
		batch_size_  = input->n();

		if (output_ == nullptr)
			output_ = new Tensor(input->shape());
		else // reshape data if cannot make up a full batch
			std::cout << "Softmax: output_ already allocated, may need reshape" << std::endl;

		if(batch_max_ == nullptr)
		{
			batch_max_ = new Tensor(input->n(), 1, 1, 1);
		}
		else
		{
			std::cout << "batch size already allocated, may need resize of reset" << std::endl;
		}
	}

	int count = input->len();
	int channels = output_->c();
	int batch_size = output_->n();

	if(input_ != output_)
	{
		cudaMemcpy(output_->get_device_ptr().get(), input_->get_device_ptr().get(), sizeof(float) * count, cudaMemcpyDefault);
	}

	kernel_channel_max<<<GET_BLOCKS(batch_size * 1), CUDA_NUM_THREADS>>>(batch_size,
			                                                             channels,
			                                                             1,
			                                                             output_->get_device_ptr().get(),
			                                                             batch_max_->get_device_ptr().get());

	kernel_channel_subtract<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, batch_size, channels, 1,
			                                                         batch_max_->get_device_ptr().get(),
			                                                         output_->get_device_ptr().get());


	kernel_exp<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count,
			                                            output_->get_device_ptr().get(),
			                                            output_->get_device_ptr().get());


	kernel_channel_sum<<<GET_BLOCKS(batch_size * 1), CUDA_NUM_THREADS>>>(batch_size,
			                                                             channels,
			                                                             1,
			                                                             output_->get_device_ptr().get(),
			                                                             batch_max_->get_device_ptr().get());


	kernel_channel_div<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, batch_size, channels,1,
			                                                    batch_max_->get_device_ptr().get(),
			                                                    output_->get_device_ptr().get());

	return output_;
}

Tensor* Softmax::backward(Tensor* target)
{
	cudaDeviceSynchronize();

	if (grad_input_ == nullptr || batch_size_ != target->n())
	{
		if (grad_input_ == nullptr)
			grad_input_ = new Tensor(input_->shape());
		else
			std::cout << "Softmax:grad_input_  already allocated, may need reshape" << std::endl;
	}

	// set grad_input_ as predict
	if(output_ != nullptr)
		cudaMemcpyAsync(grad_input_->get_device_ptr().get(),
			        	output_->get_device_ptr().get(),
			        	output_->buf_size(),
			        	cudaMemcpyDeviceToDevice);
	else
		std::cout << "Error: Softmax Backward output_ should has value, but is nullptr" << std::endl;

	// set grad_input_ = predict - target
	int count = grad_input_->len();
	kernel_saxpy<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, -1, target->get_device_ptr().get(), grad_input_->get_device_ptr().get());

	// normalize the grad_output by the batch size
	int grad_output_size = target->n() * target->c() * target->h() * target->w();
	float scale = 1.f / static_cast<float>(target->n());
	kernel_sscale<<<GET_BLOCKS(grad_output_size), CUDA_NUM_THREADS>>>(grad_output_size, scale, grad_input_->get_device_ptr().get());

	return grad_input_;
}

float Softmax::get_loss(Tensor* target)
{
	if(output_!= nullptr)
		return loss_.loss(output_, target);
	else
	{
		std::cout << "Error: Softmax::get_loss output_ should not be nullptr" << std::endl;
		return 0.0f;
	}
}

int Softmax::get_accuracy(Tensor* target)
{
	return 0;
}

