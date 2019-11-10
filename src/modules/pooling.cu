#include "pooling.h"
#include "device_util.h"
#include <cmath>
#include <cfloat>

__global__ void kernel_MaxPoolForward(int nthreads,
    float* d_bottom_data,
    int num, int channels, int height, int width,
    int pooled_height, int pooled_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float* d_top_data, float* d_mask)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		const int pw = index % pooled_width;
		const int ph = (index / pooled_width) % pooled_height;
		const int c = (index / pooled_width / pooled_height) % channels;
		const int n = index / pooled_width / pooled_height / channels;

		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		const int hend = min(hstart + kernel_h, height);
		const int wend = min(wstart + kernel_w, width);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);

		float maxval = -FLT_MAX;
		int maxidx = -1;
		const float* const bottom_slice =
			d_bottom_data + (n * channels + c) * height * width;

		for (int h = hstart; h < hend; ++h)
		{
		  for (int w = wstart; w < wend; ++w)
		  {
			if (bottom_slice[h * width + w] > maxval)
			{
				maxidx = h * width + w;
				maxval = bottom_slice[maxidx];
			}
		  }
		}
		d_top_data[index] = maxval;
		d_mask[index] = maxidx;

	}
}

__global__ void kernel_MaxPoolBackward(int nthreads,
	float* d_top_diff, float* d_top_mask,
    int num, int channels, int height, int width,
    int pooled_height, int pooled_width,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    float* d_bottom_diff)
{
	CUDA_KERNEL_LOOP(index, nthreads)
	{
		// find out the local index
		// find out the local offset
		const int w = index % width;
		const int h = (index / width) % height;
		const int c = (index / width / height) % channels;
		const int n = index / width / height / channels;
		const int phstart =
			 (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
		const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
		const int pwstart =
			 (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
		const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
		float gradient = 0;
		const int offset = (n * channels + c) * pooled_height * pooled_width;
		const float* const top_diff_slice = d_top_diff + offset;

		const float* const top_mask_slice = d_top_mask + offset;
		for (int ph = phstart; ph < phend; ++ph)
		{
			for (int pw = pwstart; pw < pwend; ++pw)
			{
			  if (top_mask_slice[ph * pooled_width + pw] == h * width + w)
			  {
				gradient += top_diff_slice[ph * pooled_width + pw];
			  }
			}
		  }

		d_bottom_diff[index] = gradient;
	  }
}

Pooling::Pooling(std::string name,
	int kernel_size,
	int padding,
	int stride):
		kernel_size_(kernel_size),
		padding_(padding),
		stride_(stride)
{
	name_ = name;
}

Pooling::~Pooling()
{
	if(mask_ != nullptr) delete mask_;
}

Tensor* Pooling::forward(Tensor* input)
{
	if (input_ == nullptr || batch_size_ != input->n())
	{
		input_ = input;

		// resource initialize
		batch_size_  = input->n();

		// setting output shape
		int pooled_height_ = static_cast<int>(ceil(static_cast<float>(
			        input_->h() + 2 * padding_ - kernel_size_) / stride_)) + 1;
		int pooled_width_ = static_cast<int>(ceil(static_cast<float>(
					input_->w() + 2 * padding_ - kernel_size_) / stride_)) + 1;

		output_size_ = Shape(input_->n(), input_->c(), pooled_height_, pooled_width_);

		std::cout << "output shape: " << output_size_.n_ << output_size_.c_ << output_size_.h_ << output_size_.w_ << std::endl;


		if (output_ == nullptr)
			output_ = new Tensor(output_size_);
		else
		{
			std::cout << "Pooling-forward output_ already exist, need reset" << std::endl;
			//output_->reset(output_size_);
		}

		if (mask_ == nullptr)
			mask_ = new Tensor(output_size_);
		else
		{
			std::cout << "Pooling-forward mask already exist, need reset" << std::endl;
			//mask_->reset(output_size_);
		}

	}

	//cudnnPoolingForward(cuda_->cudnn(), pool_desc_,&cuda_->one,   input_desc_,  input_->cuda(), &cuda_->zero,  output_desc_, output_->cuda());
	int count = output_->len();
	kernel_MaxPoolForward<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
	        count, input_->get_device_ptr().get(),
	        input_->n(), input_->c(),input_->h(), input_->w(),
	        output_size_.h(), output_size_.w(),
	        kernel_size_, kernel_size_,
	        stride_, stride_,
	        padding_, padding_,
	        output_->get_device_ptr().get(),
	        mask_->get_device_ptr().get());


	mask_->print_tensor("mask_", true, 1, 4);

	return output_;
}

Tensor* Pooling::backward(Tensor* grad_output)
{
	if (grad_input_ == nullptr || batch_size_ != grad_output->n())
	{
		grad_output_ = grad_output;

		if (grad_input_ == nullptr)
			grad_input_ = new Tensor(input_->shape());
		else
		{
			std::cout << "Pooling-backward grad_input_ already exist, need reset" << std::endl;
			//grad_input_->reset(input_->shape());
		}
	}

	int count = input_->len();

	kernel_MaxPoolBackward<<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count,
			grad_output_->get_device_ptr().get(), mask_->get_device_ptr().get(),
	    output_->n(), output_->c(), input_->h(), input_->w(),
	    output_size_.h(), output_size_.w(),
	    kernel_size_, kernel_size_,
	    stride_, stride_,
	    padding_, padding_,
	    grad_input_->get_device_ptr().get());


	return grad_input_;
}



