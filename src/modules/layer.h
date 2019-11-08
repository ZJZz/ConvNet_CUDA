#ifndef _LAYER_H_
#define _LAYER_H_

#include "../tensor.h"
#include <cuda_runtime.h>
#include <random>


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
		virtual float get_loss(Tensor *target) { return 0.0f; }
		virtual int get_accuracy(Tensor *target) { return 0; }

		// weight freeze or unfreeze
		void set_gradient_stop() { gradient_stop_ = true; }
		void freeze()   { freeze_ = true; }
		void unfreeze() { freeze_ = false; }

		void init_weight_bias(unsigned int seed = 0);
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






#endif // _LAYER_H_
