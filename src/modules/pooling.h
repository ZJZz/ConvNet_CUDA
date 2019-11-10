#include "layer.h"

class Pooling: public NNLayer
{
public:
    Pooling(std::string name,
            int kernel_size,
            int padding,
            int stride);
    ~Pooling();

    Tensor* forward(Tensor* input);
    Tensor* backward(Tensor* grad_output);

private:



    int kernel_size_;
    int padding_;
    int stride_;

    Tensor* mask_ = nullptr;
    Shape output_size_;

};
