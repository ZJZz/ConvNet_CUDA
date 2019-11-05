#ifndef _NETWORK_H_
#define _NETWORK_H_

#include "loss/cross_entropy.h"
#include "modules/layer.h"
#include "tensor.h"
#include <vector>

class Network
{
public:
    Network();
    ~Network();

    void add_layer(NNLayer *layer);

    Tensor *forward(Tensor *input);
    void backward(Tensor *input = nullptr);
    void update(float learning_rate = 0.02f);

    float loss(Tensor *target);
    int get_accuracy(Tensor *target);

    void train();
    void test();

    Tensor *output_ = nullptr;

    std::vector<NNLayer *> layers() { return layers_; }


 private:
    std::vector<NNLayer *> layers_;

};


#endif // _NETWORK_H_
