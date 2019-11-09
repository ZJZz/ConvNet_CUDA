#ifndef _TENSOR_H_
#define _TENSOR_H_

#include "shape.h"
#include <memory>
#include <iostream>
#include <cstdio>
#include <array>
#include <vector>

class Tensor
{
    public:
        // ctor
        Tensor(int batch_num = 1, int channel = 1, int height = 1, int width = 1, std::vector<float> v = {});
        Tensor(Shape shape, std::vector<float> v = {});

        // member function
        void allocateMemory();
        void allocateMemoryIfNotAllocated(Shape shape);

        void transfer_H2D();
        void transfer_D2H();

        // Tensor Dimension
       Shape shape()
        {
        	return shape_;
        }

        int n() const { return shape_.n_; }
        int c() const { return shape_.c_; }
        int h() const { return shape_.h_; }
        int w() const { return shape_.w_; }

        // return number of elements for 1 batch
        int size() { return shape_.size_per_batch(); }

        // return number of total elements in tensor including batch
        int len() { return shape_.total_elements(); }

        // return size of allocated memory
        int buf_size() { return sizeof(float) * len(); }

        std::shared_ptr<float> get_host_ptr() {
			if(host_allocated_) return h_data_;
			else return nullptr;
        }

        std::shared_ptr<float> get_device_ptr()
		{
			if(device_allocated_) return d_data_;
			else return nullptr;
		}

        // TODO:
        void reset(int n = 1, int c = 1, int h = 1, int w = 1);
        void reset(Shape shape);

        void print_tensor(std::string name, bool view_param, int num_batch, int width);

    	float& operator[](const int index);
    	const float& operator[](const int index) const;

    private:

    	// member function
        void allocateCudaMemory();
        void allocateHostMemory();

        // data member
        Shape shape_;

        std::shared_ptr<float> d_data_;
        std::shared_ptr<float> h_data_;

        bool device_allocated_;
        bool host_allocated_;

        std::vector<float> init_vec_;

};

#endif // _TENSOR_H_
