#ifndef _RELU_H_
#define _RELU_H_

#include "layer.h"

class ReLU: public NNLayer
{
public:
	ReLU(std::string name);
	~ReLU();

	Tensor* forward(Tensor* input);
	Tensor* backward(Tensor* grad_pre);

};

#endif // _RELU_H_
