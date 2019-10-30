#include <iostream>
#include "src/shape.h"
#include "src/tensor.h"
#include "src/mnist.h"

using namespace std;

int main(int argc, char* argv[])
{

	int batch_size_train = 256;

    /* Welcome Message */
    std::cout << "== MNIST training with CUDA ==" << std::endl;

    // phase 1. training
    std::cout << "[TRAIN]" << std::endl;

    // step 1. loading dataset
    MNIST train_data_loader = MNIST("./dataset");
    train_data_loader.train(batch_size_train, true);

    // TODO: step 2. create model


    // step 3. train
    int step = 0;
    Tensor* train_data = train_data_loader.get_data();
    Tensor* train_target = train_data_loader.get_target();
    train_data_loader.get_batch();

    std::cout << "load dataset success !" << std::endl;


	return 0;
}
