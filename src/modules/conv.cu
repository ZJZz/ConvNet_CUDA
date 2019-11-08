#include "conv.h"
#include "device_util.h"
#include <vector>


#define BLOCK_DIM_1D    512
#define TILE_DIM        16

__global__ void kernel_im2col(int n, float* data_im,
    int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h,  int stride_w,
    int height_col,  int width_col,
    float* data_col)
{

	CUDA_KERNEL_LOOP(index, n)
	{
		//printf("threadIdx.x: %d\n", threadIdx.x);
		int h_index = index / width_col;

		int h_col = h_index % height_col;
		int w_col = index % width_col;

		int c_im = h_index / height_col;
		int c_col = c_im * kernel_h * kernel_w;

		int h_offset = h_col * stride_h - pad_h;
		int w_offset = w_col * stride_w - pad_w;

		float* data_col_ptr = data_col;
		data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;


		const float* data_im_ptr = data_im;
		data_im_ptr += (c_im * height + h_offset) * width + w_offset;



		for (int i = 0; i < kernel_h; ++i)
		{
			for (int j = 0; j < kernel_w; ++j)
			{
				int h_im = h_offset + i;
				int w_im = w_offset + j;



				*data_col_ptr = (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ? data_im_ptr[i * width + j] : 0;
				// printf("data_col_ptr_[%d] <- data_im_ptr_index[%d]\n", (c_col * height_col + h_col) * width_col + w_col + (i * kernel_h + j), (c_im * height + h_offset) * width + w_offset + i * width + j);

				data_col_ptr += height_col * width_col;

			}
		}
	}
}

__global__ void kernel_MatMul_conv(float* A, float* B, float* C, int ARows, int ACols, int BRows,
    int BCols, int CRows, int CCols, bool add)
{
    float CValue = 0;

    int Row = blockIdx.y*TILE_DIM + threadIdx.y;
    int Col = blockIdx.x*TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (TILE_DIM + ACols - 1)/TILE_DIM; k++) {

         if (k*TILE_DIM + threadIdx.x < ACols && Row < ARows)
             As[threadIdx.y][threadIdx.x] = A[Row*ACols + k*TILE_DIM + threadIdx.x];
         else
             As[threadIdx.y][threadIdx.x] = 0.0;

         if (k*TILE_DIM + threadIdx.y < BRows && Col < BCols)
             Bs[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*BCols + Col];
         else
             Bs[threadIdx.y][threadIdx.x] = 0.0;

         __syncthreads();

         for (int n = 0; n < TILE_DIM; ++n)
             CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

         __syncthreads();
    }

    if (Row < CRows && Col < CCols)
    {
    	if(!add)
    		C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
    		           (blockIdx.x * blockDim.x)+ threadIdx.x] = CValue;
    	else
    		C[((blockIdx.y * blockDim.y + threadIdx.y)*CCols) +
    		    		           (blockIdx.x * blockDim.x)+ threadIdx.x] += CValue;

    }

}

__global__ void kernel_init_one_vec_conv(float* d_one_vec, size_t length)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= length) return;

	d_one_vec[i] = 1.f;
}

Conv2D::Conv2D(std::string name,
				int out_channels,
				int kernel_size,
				int stride,
				int padding):
				out_channels_(out_channels),
				kernel_size_(kernel_size),
				stride_(stride),
				padding_(padding)
{
	name_ = name;
}

Conv2D::~Conv2D()
{
	if (d_one_vec_ != nullptr)
		cudaFree(d_one_vec_);
	if(col_buffer_ != nullptr)
		delete col_buffer_;
}

Tensor* Conv2D::forward(Tensor* input)
{
	// initialize weights and bias
	if (weights_ == nullptr)
	{
		// initialize containers
		// may need  transpose

		//std::vector<float> w_v = {1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0,
		//                     1.0, 0.0, 0.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 0.0};

		//std::vector<float> w_v = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

		weights_ = new Tensor(out_channels_, input->c(), kernel_size_, kernel_size_);

		//weights_->print_tensor("weight", true, 2, 12);


		std::vector<float> b_v = {1.0, 1.0};
		biases_  = new Tensor(1,1, 1, out_channels_, b_v);	// bias size
		//bias_desc_ = biases_->tensor();
	}

	if (input_ == nullptr || batch_size_ != input->n())
	{
		// initialize input
		input_ = input;
		//input_desc_ = input->tensor();
		batch_size_  = input->n();

		// initialize output
		compute_output_shape();

		if (output_ == nullptr)
			output_  = new Tensor(output_size_);
		else
		{
			//output_->reset(output_size_);
			std::cout << "output already exist, need reset" << std::endl;
		}

		ouput_spatial_dim_ =  output_size_.h_ * output_size_.w_;

		//output_desc_ = output_->tensor();

		// initialize workspace for cudnn
		//set_workspace();

		if (!freeze_)
		{
			//std::cout << "Test: uncomment it later" << std::endl;
			init_weight_bias();
		}
	}

	//获取一个输出通道对应的所有卷积核对输入的一个卷积组所有通道操作一次处理数据量大小，为(输入总通道数)*卷积核高*卷积核宽
	kernel_dim_ = input_->c() *  kernel_size_ *  kernel_size_;
	//weight_offset_ = out_channels_ * kernel_dim_;

	if(col_buffer_ == nullptr)
	{
		// may need transpose
		col_buffer_ = new Tensor(1, 1, kernel_dim_, output_size_.h_ * output_size_.w_);
	}
	// Conv
	float* weight_dev = weights_->get_device_ptr().get();
	float* input_dev = input_->get_device_ptr().get();
	float* output_dev = output_->get_device_ptr().get();

	for (int n = 0; n < batch_size_; ++n)
	{
	  // input * weight gemm
	  forward_gemm(input_dev + n * input_->size(), weight_dev,
			  output_dev + n * output_->size());

	  float* bias = biases_->get_device_ptr().get();
	  // bias gemm
	  this->forward_bias(output_dev + n * output_->size(), bias);
	}

	return output_;
}

Tensor* Conv2D::backward(Tensor* grad_output)
{
	//TODO:
	return nullptr;
}



void Conv2D::compute_output_shape()
{
	//在这里计算卷积过后生成的Tensor的Shape
	int output_dim = (input_->h() + 2 * padding_ - kernel_size_) / stride_ + 1;

	// may need transpose
	output_size_ = Shape(batch_size_, out_channels_, output_dim, output_dim);
}


void Conv2D::forward_gemm(float* input, float* weights, float* output)
{
	// im2col
	float* col_buff = input;
	conv_im2col_wraper(input, col_buffer_->get_device_ptr().get());
	col_buff = col_buffer_->get_device_ptr().get();

	col_buffer_->print_tensor("col_buffer", true, 1, 4);

	// /gemm with weight
	dim3 threads, grid;
	threads = dim3(TILE_DIM, TILE_DIM);
	grid = dim3(( ouput_spatial_dim_ + threads.x - 1) / threads.x, (out_channels_ + threads.y - 1) / threads.y);

	kernel_MatMul_conv<<<grid, threads>>>(weights, col_buff, output, out_channels_ ,kernel_dim_, kernel_dim_, ouput_spatial_dim_ ,
			 out_channels_, ouput_spatial_dim_, false);



}

void Conv2D::forward_bias(float* output, float* bias)
{
	if (d_one_vec_ != nullptr) cudaFree(d_one_vec_);

	cudaMalloc((void**)&d_one_vec_, sizeof(float) * ouput_spatial_dim_);
	kernel_init_one_vec_conv<<< (ouput_spatial_dim_+BLOCK_DIM_1D-1)/BLOCK_DIM_1D, BLOCK_DIM_1D >>>(d_one_vec_, ouput_spatial_dim_);

	dim3 threads, grid;
	threads = dim3(TILE_DIM, TILE_DIM);
	grid = dim3(( ouput_spatial_dim_ + threads.x - 1) / threads.x, (out_channels_ + threads.y - 1) / threads.y);

	kernel_MatMul_conv<<<grid, threads>>>(bias, d_one_vec_, output, out_channels_, 1, 1,
			ouput_spatial_dim_, out_channels_, ouput_spatial_dim_, true);
}

inline void Conv2D::conv_im2col_wraper(float* data, float* col_buff)
{
	conv_im2col(data, input_->c(), input_->h(), input_->w(),
			    kernel_size_, kernel_size_,
			    padding_, padding_,
			    stride_, stride_,
			    col_buff);
}

void Conv2D::conv_im2col(float* data_im, int channels,
    		         int height, int width,
    		         int kernel_h, int kernel_w,
    		         int pad_h, int pad_w,
    		         int stride_h, int stride_w,
    		         float* data_col)
{

//	std::cout << "channels: " << channels << std::endl;
//	std::cout << "height: " <<  height << std::endl;
//	std::cout << "width: " << width << std::endl;
//	std::cout << "kernel_h: " <<kernel_h << std::endl;
//	std::cout << "kernel_w: " << kernel_w << std::endl;
//	std::cout << "pad_h: " << pad_h << std::endl;
//	std::cout << "pad_w: " << pad_w << std::endl;
//	std::cout << "stride_h: " << stride_h << std::endl;
//	std::cout << "stride_w: " << stride_w << std::endl;


	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad_h - (kernel_h - 1) - 1) / stride_h + 1;
	int width_col = (width + 2 * pad_w - (kernel_w - 1) - 1) / stride_w + 1;
	int num_kernels = channels * height_col * width_col;

//	std::cout << "height_col: " << height_col << std::endl;
//	std::cout << "width_col: " << width_col << std::endl;
//	std::cout << "num_kernels: " << num_kernels << std::endl;

	kernel_im2col<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
	  num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
	  pad_w, stride_h, stride_w, height_col, width_col, data_col);
}


