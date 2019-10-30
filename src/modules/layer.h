#ifndef _LAYER_H_
#define _LAYER_H_

#include "tensor.h"

class NNLayer
{
	public:
		virtual ~NNLayer() = 0;

		virtual Tensor* forward(Tensor* A) = 0;
		virtual Tensor* backprop(Tensor* dZ, float learning_rate) = 0;

		std::string get_name() { return name_; };

	protected:
		std::string name_;

};

inline NNLayer::~NNLayer() {}

#endif // _LAYER_H_
