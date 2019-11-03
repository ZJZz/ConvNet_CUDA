#ifndef _SOFTMAX_H_
#define _SOFTMAX_H_

#include "layer.h"
#include "../loss/cross_entropy.h"

class Softmax : public NNLayer
{
public:
	Softmax(std::string name);
	~Softmax();

	Tensor* forward(Tensor* input);
	Tensor* backward(Tensor* grad_input);

	float get_loss(Tensor* target);
	int get_accuracy(Tensor* target);

private:
	Tensor* batch_max_ = nullptr;
	CrossEntropyLoss loss_;

};

#endif // _SOFTMAX_H_
