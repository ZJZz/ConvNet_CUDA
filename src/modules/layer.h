#ifndef _LAYER_H_
#define _LAYER_H_

#include "../tensor.h"

class NNLayer
{
	public:
		NNLayer() {} ;
		virtual ~NNLayer() {
			if (output_       != nullptr)  delete output_;
			if (grad_input_   != nullptr)  delete grad_input_;

			if (weights_      != nullptr)  delete weights_;
			if (biases_       != nullptr)  delete biases_;
			if (grad_weights_ != nullptr)  delete grad_weights_;
			if (grad_biases_  != nullptr)  delete grad_biases_;
		};

		virtual Tensor* forward(Tensor* A) = 0;
		virtual Tensor* backward(Tensor* dZ) = 0;

		std::string get_name() { return name_; };

		// only used in last layer
		virtual float get_loss(Tensor *target) {};
		virtual int get_accuracy(Tensor *target) {};

		// weight freeze or unfreeze
		void set_gradient_stop() { gradient_stop_ = true; }
		void freeze()   { freeze_ = true; }
		void unfreeze() { freeze_ = false; }

		void init_weight_bias(unsigned int seed);
		void update_weights_biases(float learning_rate);




	protected:
		std::string name_;

		// control parameter updates
		bool freeze_ = false;

		// gradient stop tagging
		bool gradient_stop_ = false;

		// output tensor
		Tensor* input_ = nullptr; // x: layer input
		Tensor* output_ = nullptr; // y: layer output
		Tensor* grad_input_ = nullptr; // dx: layer input's gradient
		Tensor* grad_output_ = nullptr; // dy: layer output's gradient

		int batch_size_= 0;

		Tensor* weights_ = nullptr; // w
		Tensor* biases_ = nullptr; // b
		Tensor* grad_weights_ = nullptr; // dw
		Tensor* grad_biases_ = nullptr; // db

		friend class Network;
};

void NNLayer::init_weight_bias(unsigned int seed)
{
	checkCudaErrors(cudaDeviceSynchronize());

	if (weights_ == nullptr || biases_ == nullptr)
		return;

	// Create random network
	std::random_device rd;
	std::mt19937 gen(seed == 0 ? rd() : static_cast<unsigned int>(seed));

	// He uniform distribution
	float range = sqrt(6.f / input_->size());	// He's initialization
	std::uniform_real_distribution<> dis(-range, range);

	for (int i = 0; i < weights_->len(); i++)
		weights_->ptr()[i] = static_cast<float>(dis(gen));
	for (int i = 0; i < biases_->len(); i++)
		biases_->ptr()[i] = 0.f;

	// copy initialized value to the device
	weights_->to(DeviceType::cuda);
	biases_->to(DeviceType::cuda);

	std::cout << ".. initialized " << name_ << " layer .." << std::endl;
}


// TODO: replace cublasSaxpy with my kernel
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
		checkCublasErrors(
			cublasSaxpy(cuda_->cublas(),
				weights_->len(),
				&eps,
				grad_weights_->cuda(), 1,
				weights_->cuda(), 1));

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
		checkCublasErrors(
			cublasSaxpy(cuda_->cublas(),
				biases_->len(),
				&eps,
				grad_biases_->cuda(), 1,
				biases_->cuda(), 1));

#if (DEBUG_UPDATE)
		biases_->print(name_ + "biases (after update)", true);
		// getchar();
#endif // DEBUG_UPDATE
	}
}

#endif // _LAYER_H_
