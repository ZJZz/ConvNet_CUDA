#include <iostream>
#include "src/shape.h"
#include "src/tensor.h"
#include "src/mnist.h"
#include "src/modules/relu.h"
#include "src/modules/softmax.h"

using namespace std;

int main(int argc, char* argv[])
{

//	int batch_size_train = 256;
//
//    /* Welcome Message */
//    std::cout << "== MNIST training with CUDA ==" << std::endl;
//
//    // phase 1. training
//    std::cout << "[TRAIN]" << std::endl;
//
//    // step 1. loading dataset
//    MNIST train_data_loader = MNIST("./dataset");
//    train_data_loader.train(batch_size_train, true);
//
//    // TODO: step 2. create model
//
//
//    // step 3. train
//    int step = 0;
//    Tensor* train_data = train_data_loader.get_data();
//    Tensor* train_target = train_data_loader.get_target();
//    train_data_loader.get_batch();
//
//    std::cout << "load dataset success !" << std::endl;
	// batch_size = 1
	Shape s1 = Shape(1,10,1,1);
	vector<float> v1    = {0.01, 0.02, 0.03, 0.04, 0.999, 0.01, 0.12, 0.14, 0.16, 0.23};
	Tensor* t1 = new Tensor(s1, v1);

	vector<float> label1 = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
	Tensor* target1 = new Tensor(s1, label1);

	// batch_size = 2
	Shape s2 = Shape(2,10,1,1);
	vector<float> v2    = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
						   0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};

	vector<float> v3    = {0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
						   0.12, 0.44, 1.7, 0.6, 0.95, 0.4, 0.3, 0.2, 0.1, 0};
	Tensor* t2 = new Tensor(s2, v3);

	vector<float> label2 = {0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			               0, 0, 0, 0, 0, 0, 0, 0, 1, 0,};
	Tensor* target2 = new Tensor(s2, label2);


    // ReLU Test
//	ReLU* relu = new ReLU("relu_1");
//	t = relu->forward(t);
//	t->print_tensor("relu_1", true, 1, 3);

	Softmax* softmax = new Softmax("softmax_1");
	t1 = softmax->forward(t1);
	t1->print_tensor("softmax_forward", true, 1, 10);

	std::cout << "loss: " << softmax->get_loss(target1) << std::endl;

//	t = softmax->backward(target);
//	t->print_tensor("softmax_backward", true, 2, 10);



	return 0;
}
