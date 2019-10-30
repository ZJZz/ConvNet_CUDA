#include "tensor.h"
#include <cuda_runtime.h>

Tensor::Tensor(int batch_num, int channel, int height, int width):
    shape_(batch_num, channel, height, width),d_data_(nullptr), h_data_(nullptr),device_allocated_(false), host_allocated_(false)
{
	allocateMemoryIfNotAllocated(shape_);
}

Tensor::Tensor(Shape shape):Tensor(shape.n_, shape.c_, shape.h_, shape.w_)
{ }


void Tensor::allocateMemoryIfNotAllocated(Shape& shape)
{
    if(!device_allocated_ && !host_allocated_)
    {
        shape_ = shape;
        allocateMemory();
    }
}

void Tensor::allocateMemory()
{
    allocateHostMemory();
    allocateCudaMemory();
}

void Tensor::allocateHostMemory()
{
    if(!host_allocated_)
    {
        h_data_ = std::shared_ptr<float>(new float[shape_.total_elements()], [&](float *ptr){  delete[] ptr; });
        host_allocated_ = true;
    }
}

void Tensor::allocateCudaMemory()
{
    if(!device_allocated_)
    {
        float *device_mem = nullptr;
        cudaMalloc(&device_mem, shape_.total_elements() * sizeof(float));
        // TODO: add some exception handle
        d_data_ = std::shared_ptr<float>(device_mem, [&](float *ptr){ cudaFree(ptr); });
        device_allocated_ = true;
    }
}

void Tensor::transfer_H2D()
{
    if(device_allocated_ && host_allocated_)
    {
        // get() ?
        cudaMemcpy(h_data_.get(), d_data_.get(), shape_.total_elements() * sizeof(float), cudaMemcpyHostToDevice);
        // TODO: add some exception handle
    }
    else
    {
        printf("Cannot copy host data to not allocated memory on device\n");
    }
}

void Tensor::transfer_D2H()
{
    if(device_allocated_ && host_allocated_)
    {
        // get() ?
        cudaMemcpy(h_data_.get(), d_data_.get(), shape_.total_elements() * sizeof(float), cudaMemcpyDeviceToHost);
        // TODO: add some exception handle
    }
    else
    {
        printf("Cannot copy device data to not allocated memory on host\n");
    }
}

float& Tensor::operator[](const int index)
{
	return h_data_.get()[index];
}

const float& Tensor::operator[](const int index) const
{
	return h_data_.get()[index];
}

void Tensor::print_tensor(std::string name, bool view_param = false, int num_batch = 1, int width = 16)
{
	transfer_D2H();
	std::cout << "**" << name << "\t: (" << size() << ")\t";
	std::cout << "n: " << shape_.n_ << " c: " << shape_.c_ << " h: " << shape_.h_ << " w: " << shape_.w_ << std::endl;
	std::cout << std::hex << "\t(host:" << h_data_ << ", device: " << d_data_ << ")" << std::dec << std::endl;

	if(view_param)
	{
		std::cout << std::fixed;
		std::cout.precision(6);

		int max_print_line = 4;
		if(width == 28)
		{
			std::cout.precision(3);
			max_print_line = 28;
		}

		int offset = 0;

		for(int n = 0; n < num_batch; n++)
		{
			if(num_batch > 1)
				std::cout << "<--- batch[" << n << "] --->" << std::endl;

			int count = 0;
			int print_line_count = 0;
			while( count < size() && print_line_count < max_print_line )
			{
				std::cout << "\t";
				for(int s = 0; s < width && count < size(); s++)
				{
					std::cout << h_data_.get()[size() * n + count + offset] << "\t";
					count++;
				}
				std::cout << std::endl;
				print_line_count++;
			}
		}

		std::cout.unsetf(std::ios::fixed);
	}
}



