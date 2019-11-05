#ifndef _LINEAR_H_
#define _LINEAR_H_

#include "layer.h"

class Linear: public NNLayer
{
public:
	Linear(std::string name, int out_size);
	~Linear();

	Tensor* forward(Tensor* input);
	Tensor* backward(Tensor* grad_input);

private:
	int input_size_ = 0;
	int output_size_= 0;

	float *d_one_vec = nullptr;
	Tensor* weights_trans_ = nullptr;
	Tensor* grad_output_trans_ = nullptr;

};

#endif //  _LINEAR_H_
