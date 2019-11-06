#include "network.h"

Network::Network()
{
}

Network::~Network()
{
	// destroy network
	for (auto layer: layers_)
		delete layer;
}

void Network::add_layer(NNLayer *layer)
{
	layers_.push_back(layer);

	// tagging layer to stop gradient if it is the first layer
	if (layers_.size() == 1)
		layers_.at(0)->set_gradient_stop();
}


Tensor *Network::forward(Tensor *input)
{
	output_ = input;

	for (auto layer : layers_)
	{
		#if (DEBUG_FORWARD)
		std::cout << "[[Forward ]][[ " << std::setw(7) << layer->get_name() << " ]]\t(" << output_->n() << ", " << output_->c() << ", " << output_->h() << ", " << output_->w() << ")\t";
		#endif // DEBUG_FORWARD

		output_ = layer->forward(output_);

		#if (DEBUG_FORWARD)
		std::cout << "--> (" << output_->n() << ", " << output_->c() << ", " << output_->h() << ", " << output_->w() << ")" << std::endl;
		checkCudaErrors(cudaDeviceSynchronize());

		#if (DEBUG_FORWARD > 1)
			output_->print("output", true);

			if (phase_ == inference)
				getchar();
		#endif
		#endif // DEBUG_FORWARD

		// TEST
		// checkCudaErrors(cudaDeviceSynchronize());
	}

	return output_;
}

void Network::backward(Tensor *target)
{
	Tensor *gradient = target;

	// back propagation.. update weights internally.....
	for (auto layer = layers_.rbegin(); layer != layers_.rend(); layer++)
	{
		// getting back propagation status with gradient size

#if (DEBUG_BACKWARD)
		std::cout << "[[Backward]][[ " << std::setw(7) << (*layer)->get_name() << " ]]\t(" << gradient->n() << ", " << gradient->c() << ", " << gradient->h() << ", " << gradient->w() << ")\t";
#endif // DEBUG_BACKWARD

		gradient = (*layer)->backward(gradient);

#if (DEBUG_BACKWARD)
		// and the gradient result
		std::cout << "--> (" << gradient->n() << ", " << gradient->c() << ", " << gradient->h() << ", " << gradient->w() << ")" << std::endl;
		checkCudaErrors(cudaDeviceSynchronize());

#if (DEBUG_BACKWARD > 1)
		gradient->print((*layer)->get_name() + "::dx", true);
		getchar();
#endif
#endif // DEBUG_BACKWARD
	}
}

void Network::update(float learning_rate)
{

#if (DEBUG_UPDATE)
	std::cout << "Start update.. lr = " << learning_rate << std::endl;
#endif

	for (auto layer : layers_)
	{
		// if no parameters, then pass
		if (layer->weights_ == nullptr || layer->grad_weights_ == nullptr ||
			layer->biases_ == nullptr || layer->grad_biases_ == nullptr)
			continue;

		layer->update_weights_biases(learning_rate);
	}
}

void Network::train()
{
	// unfreeze all layers
	for (auto layer : layers_)
	{
		layer->unfreeze();
	}
}

void Network::test()
{
	// freeze all layers
	for (auto layer : layers_)
	{
		layer->freeze();
	}
}

float Network::loss(Tensor *target)
{
	NNLayer *layer = layers_.back();
	return layer->get_loss(target);
}

int Network::get_accuracy(Tensor *target)
{
	NNLayer *layer = layers_.back();
	return layer->get_accuracy(target);
}
