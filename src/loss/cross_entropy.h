#ifndef _CROSS_ENTROPY_H_
#define _CROSS_ENTROPY_H_

#include "../tensor.h"

class CrossEntropyLoss
{
public:
    CrossEntropyLoss();
    ~CrossEntropyLoss();

    float loss(Tensor *predict, Tensor *target);
    float accuracy(Tensor *predict, Tensor *target);

private:
    // reduced loss
    float h_loss_ = 0.f;
    float *d_loss_ = nullptr;

    float *d_workspace_ = nullptr;
    void init_workspace(int batch_size);
};

#endif // _CROSS_ENTROPY_H_
