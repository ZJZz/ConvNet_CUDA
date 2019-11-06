#include "layer.h"
#include "device_util.h"

__global__ void kernel_saxpy_layer(int n, float alpha, float* x, float* y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n) y[i] = alpha * x[i] + y[i];
}

void NNLayer::update_weights_biases(float learning_rate)
{
	float eps = -1.f * learning_rate;
	if (weights_ != nullptr && grad_weights_ != nullptr)
	{
#if (DEBUG_UPDATE)
		weights_->print(name_ + "::weights (before update)", true);
		grad_weights_->print(name_ + "::gweights", true);
#endif // DEBUG_UPDATE

		// w = w + eps * dw
		kernel_saxpy_layer<<<GET_BLOCKS(weights_->len()), CUDA_NUM_THREADS>>>(weights_->len(), eps,
				     grad_weights_->get_device_ptr().get(),
				     weights_->get_device_ptr().get());



#if (DEBUG_UPDATE)
		weights_->print(name_ + "weights (after update)", true);
		// getchar();
#endif // DEBUG_UPDATE
	}

	if (biases_ != nullptr && grad_biases_ != nullptr)
	{
#if (DEBUG_UPDATE)
		biases_->print(name_ + "biases (before update)", true);
		grad_biases_->print(name_ + "gbiases", true);
#endif // DEBUG_UPDATE

		// b = b + eps * db
		kernel_saxpy_layer<<<GET_BLOCKS(biases_->len()), CUDA_NUM_THREADS>>>(biases_->len(), eps,
								  grad_biases_->get_device_ptr().get(),
								  biases_->get_device_ptr().get());

#if (DEBUG_UPDATE)
		biases_->print(name_ + "biases (after update)", true);
		// getchar();
#endif // DEBUG_UPDATE
	}
}

void NNLayer::init_weight_bias(unsigned int seed)
{
	cudaDeviceSynchronize();

	if (weights_ == nullptr || biases_ == nullptr)
		return;

	// Create random network
	std::random_device rd;
	std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

	// He uniform distribution
	float range = sqrt(6.f / input_->size());	// He's initialization
	std::uniform_real_distribution<> dis(-range, range);

	for (int i = 0; i < weights_->len(); i++)
		weights_->get_host_ptr().get()[i] = static_cast<float>(dis(gen));
	for (int i = 0; i < biases_->len(); i++)
		biases_->get_host_ptr().get()[i] = 0.f;

	// copy initialized value to the device
	weights_->transfer_H2D();
	biases_->transfer_H2D();

	std::cout << ".. initialized " << name_ << " layer .." << std::endl;
}
