#include "neural_network.h"

NeuralNetwork::NeuralNetwork(float lr = 0.01) { }

NeuralNetwork::~NeuralNetwork()
{
    for(auto layer: layers_)
        delete layer;
}

Tensor NeuralNetwork::forward(Tensor X)
{
    Y_ = X;

    for(auto layer : layers_)
    {
        Y_ = layer->forward(Y_);
    }
    
    return Y_;
}

void NeuralNetwork::backward(Tensor predictions, Tensor target)
{
    dY.allocateMemoryIfNotAllocated(predictions.shape);
	Tensor error = loss.dCost(predictions, target, dY);

	for (auto it = this->layers_.rbegin(); it != this->layers_.rend(); it++) {
		error = (*it)->backward(error, learning_rate);
	}

	cudaDeviceSynchronize();
}

void NeuralNetwork::addLayer(Layer *layer)
{
    layers_.push_back(layer);
}

std::vector<Layer *> NeuralNetwork::getLayers() const
{
    return layers_;
}

void NeuralNetwork::train_mode()
{
    for(auto layer: layers_)
        layer->unfreeze();
}