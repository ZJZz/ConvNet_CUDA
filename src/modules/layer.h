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
		void freeze()   { freeze_ = true; }
		void unfreeze() { freeze_ = false; }


	protected:
		std::string name_;

		// control parameter updates
		bool freeze_ = false;

		// gradient stop tagging
		bool gradient_stop__false = false;

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
