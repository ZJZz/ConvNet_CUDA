#include "conv.h"
#include "device_util.h"
#include <vector>
#include <iostream>


#define BLOCK_DIM_1D    512
#define TILE_DIM        16
#define BLOCK_DIM       16

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


__global__ void kernel_col2im(int n, float* data_col,
    int height, int width, int channels,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int height_col, int width_col,
    float* data_im)
{
	CUDA_KERNEL_LOOP(index, n)
	{
		float val = 0;
		const int w_im = index % width + pad_w;
		const int h_im = (index / width) % height + pad_h;
		const int c_im = index / (width * height);
		int kernel_extent_w = (kernel_w - 1) * 1 + 1;
		int kernel_extent_h = (kernel_h - 1) * 1 + 1;
		// compute the start and end of the output
		const int w_col_start =
			(w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
		const int w_col_end = min(w_im / stride_w + 1, width_col);
		const int h_col_start =
			(h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
		const int h_col_end = min(h_im / stride_h + 1, height_col);
		// TODO: use LCM of stride and dilation to avoid unnecessary loops
		for (int h_col = h_col_start; h_col < h_col_end; h_col += 1)
		{
		  for (int w_col = w_col_start; w_col < w_col_end; w_col += 1)
		  {
			int h_k = (h_im - h_col * stride_h);
			int w_k = (w_im - w_col * stride_w);
			if (h_k % 1 == 0 && w_k % 1 == 0) {
			  h_k /= 1;
			  w_k /= 1;
			  int data_col_index = (((c_im * kernel_h + h_k) * kernel_w + w_k) *
									height_col + h_col) * width_col + w_col;
			  val += data_col[data_col_index];
			}
		  }
		}
		data_im[index] = val;
	}
}

__global__ void kernel_transpose_conv(float *odata, float *idata, int width, int height)
{
	__shared__ float block[TILE_DIM][TILE_DIM+1];

	// read the matrix tile into shared memory
    // load one element per thread from device memory (idata) and store it
    // in transposed order in block[][]
	unsigned int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		block[threadIdx.y][threadIdx.x] = idata[index_in];
	}

    // synchronise to ensure all writes to block[][] have completed
	__syncthreads();

	// write the transposed matrix tile to global memory (odata) in linear order
	xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
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

__global__ void kernel_MatVec_conv(float *device_Mat, float *device_Vect,int matRowSize, int vlength, float *device_ResVect, bool add)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int tindex= tidx + gridDim.x * BLOCK_DIM * tidy;


	if(tindex<matRowSize)
	{
		int i;int m= tindex * vlength;
		if(!add) device_ResVect[tindex]=0.0f;
		for(i = 0; i < vlength; i++)
			device_ResVect[tindex] +=  device_Mat[m+i] * device_Vect[i];
	}

	__syncthreads();

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

		std::vector<float> w_v = {0.0, 0.0, -1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 0.0};

		weights_ = new Tensor(out_channels_, input->c(), kernel_size_, kernel_size_, w_v);
		weights_trans_ = new Tensor(input->c(), kernel_size_, kernel_size_, out_channels_);
		//weights_->print_tensor("weight", true, 2, 12);


		std::vector<float> b_v = {0.0};
		biases_  = new Tensor(1, out_channels_,1,1, b_v);	// bias size
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

		output_spatial_dim_ =  output_size_.h_ * output_size_.w_;

		//output_desc_ = output_->tensor();

		// initialize workspace for cudnn
		//set_workspace();

		if (!freeze_)
		{
			std::cout << "Test: uncomment it later" << std::endl;
			//init_weight_bias();
		}
	}

	//获取一个输出通道对应的所有卷积核对输入的一个卷积组所有通道操作一次处理数据量大小，为(输入总通道数)*卷积核高*卷积核宽
	kernel_dim_ = input_->c() *  kernel_size_ *  kernel_size_;
	//weight_offset_ = out_channels_ * kernel_dim_;

	if(col_buffer_ == nullptr)
	{
		// may need transpose
		col_buffer_ = new Tensor(1, 1, kernel_dim_, output_size_.h_ * output_size_.w_);
		col_buffer_trans_ = new Tensor(1, 1, output_size_.h_ * output_size_.w_, kernel_dim_);
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
	// initialize grad_output back-propagation space
	if (grad_input_ == nullptr || batch_size_ != grad_output->n())
	{
		grad_output_  = grad_output;
		grad_weights_ = new Tensor(weights_->shape());

		//grad_weights_->print_tensor("grad_weights_pre",true,1,9);

		grad_biases_  = new Tensor(1, biases_->c());

		//grad_biases_->print_tensor("grad_biases_pre",true,1,1);

		if (grad_input_ == nullptr)
		{
			grad_input_ = new Tensor(input_->shape());
			// grad_input_->print_tensor("grad_input_pre",true,1,5);
		}
		else
		{
			std::cout << "grad_input_ already exist, need reset" << std::endl;
		}
	}

	float* weight_dev = weights_->get_device_ptr().get();
	float* grad_weights_dev = grad_weights_->get_device_ptr().get();
	float* grad_output_dev = grad_output_->get_device_ptr().get();

	// gradients of biases
	float* grad_biases_dev = grad_biases_->get_device_ptr().get();
	for (int n = 0; n < batch_size_; ++n)
	{
		backward_bias_gemv(grad_biases_dev, grad_output_dev + n * grad_output_->size());
	}

	// gradients of weights

	float* input_dev = input_->get_device_ptr().get();
	for(int n = 0; n < batch_size_; ++n)
	{
		backward_weight_gemm(input_dev + n * input_->size(),
							 grad_output_dev + n * grad_output_->size(),
							 grad_weights_dev);
	}
	//grad_weights_->print_tensor("grad_weights_post",true,1,9);

	// gradients of input data
	if (!gradient_stop_)
	{
		float* grad_input_dev = grad_input_->get_device_ptr().get();

		// grad_output_->print_tensor("backward grad_output", true, 1, 3);

		for(int n = 0; n < batch_size_; ++n)
		{
			backward_input_gemm(grad_output_dev + n * grad_output_->size(),
								weight_dev,
								grad_input_dev + n * grad_input_->size());
		}

		// grad_input_->print_tensor("backward grad_input after", true, 1, 5);

	}

	return grad_input_;
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

	// col_buffer_->print_tensor("col_buffer", true, 1, 9);

	// gemm with weight
	dim3 threads, grid;
	threads = dim3(TILE_DIM, TILE_DIM);
	grid = dim3(( output_spatial_dim_ + threads.x - 1) / threads.x, (out_channels_ + threads.y - 1) / threads.y);

	kernel_MatMul_conv<<<grid, threads>>>(weights, col_buff, output, out_channels_ ,kernel_dim_, kernel_dim_, output_spatial_dim_ ,
			 out_channels_, output_spatial_dim_, false);



}

void Conv2D::forward_bias(float* output, float* bias)
{
	if (d_one_vec_ != nullptr) cudaFree(d_one_vec_);

	cudaMalloc((void**)&d_one_vec_, sizeof(float) * output_spatial_dim_);
	kernel_init_one_vec_conv<<< (output_spatial_dim_+BLOCK_DIM_1D-1)/BLOCK_DIM_1D, BLOCK_DIM_1D >>>(d_one_vec_, output_spatial_dim_);

	dim3 threads, grid;
	threads = dim3(TILE_DIM, TILE_DIM);
	grid = dim3(( output_spatial_dim_ + threads.x - 1) / threads.x, (out_channels_ + threads.y - 1) / threads.y);

	kernel_MatMul_conv<<<grid, threads>>>(bias, d_one_vec_, output, out_channels_, 1, 1,
			output_spatial_dim_, out_channels_, output_spatial_dim_, true);
}

inline void Conv2D::conv_im2col_wraper(float* data, float* col_buff)
{
	conv_im2col(data, input_->c(), input_->h(), input_->w(),
			    kernel_size_, kernel_size_,
			    padding_, padding_,
			    stride_, stride_,
			    col_buff);
}

inline void Conv2D::conv_col2im_wraper(float* col_buff, float* data)
{
	conv_col2im(col_buff, input_->c(), input_->h(), input_->w(),
				    kernel_size_, kernel_size_,
				    padding_, padding_,
				    stride_, stride_,
				    data);
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

void Conv2D::conv_col2im(float* data_col, int channels, int height, int width,
    		         int kernel_h, int kernel_w, int pad_h, int pad_w,
    		         int stride_h, int stride_w, float* data_im)
{
	  int height_col = (height + 2 * pad_h - (kernel_h - 1) - 1) / stride_h + 1;
	  int width_col = (width + 2 * pad_w - (kernel_w - 1) - 1) / stride_w + 1;
	  int num_kernels = channels * height * width;
	  // To avoid involving atomic operations, we will launch one kernel per
	  // bottom dimension, and then in the kernel add up the top dimensions.
	  // NOLINT_NEXT_LINE(whitespace/operators)
	  kernel_col2im<<<GET_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
	      num_kernels, data_col, height, width, channels, kernel_h, kernel_w,
	      pad_h, pad_w, stride_h, stride_w, height_col, width_col, data_im);
}

void Conv2D::backward_weight_gemm(float* input, float* output, float* weights)
{
	float* col_buff = input;
	conv_im2col_wraper(input, col_buffer_->get_device_ptr().get());
	col_buff = col_buffer_->get_device_ptr().get();


	dim3 threads, grid;
	// transpose col_buff
	threads = dim3(TILE_DIM, TILE_DIM);
	grid = dim3((output_spatial_dim_ + threads.x - 1) / threads.x, (kernel_dim_ + threads.y - 1) / threads.y );
	kernel_transpose_conv<<< grid, threads>>>(col_buffer_trans_->get_device_ptr().get(), col_buff, output_spatial_dim_, kernel_dim_);

	//col_buffer_trans_->print_tensor("col_buff_trans",true,1,9);
	// gemm
	threads = dim3(TILE_DIM, TILE_DIM);
	grid = dim3(( kernel_dim_ + threads.x - 1) / threads.x, (out_channels_ + threads.y - 1) / threads.y);

	kernel_MatMul_conv<<<grid, threads>>>(output, col_buffer_trans_->get_device_ptr().get(), weights, out_channels_, output_spatial_dim_, output_spatial_dim_, kernel_dim_,
			out_channels_, kernel_dim_, true);

}

void Conv2D::backward_input_gemm(float* output, float* weights, float* input)
{
	// need reset, or it will has bug

	if(col_buffer_ != nullptr)
	{
		// col_buffer_->reset(col_buffer_->shape());
		for(int i = 0; i < col_buffer_->len(); i++)
			col_buffer_->get_host_ptr().get()[i] = 0.0f;

		col_buffer_->transfer_H2D();
	}

	// col_buffer_->print_tensor("col_buffer_ before", true, 1, 9);
	float* col_buff = col_buffer_->get_device_ptr().get();

	dim3 threads, grid;
	// transpose weight
	threads = dim3(TILE_DIM, TILE_DIM);
	grid = dim3((kernel_dim_ + threads.x - 1) / threads.x, (out_channels_ + threads.y - 1) / threads.y );
	kernel_transpose_conv<<< grid, threads>>>(weights_trans_->get_device_ptr().get(), weights, kernel_dim_, out_channels_);

	// weights_trans_->print_tensor("backward w_Trans", true, 1, 1);

	// gemm
	threads = dim3(TILE_DIM, TILE_DIM);
	grid = dim3(( kernel_dim_ + threads.x - 1) / threads.x, (out_channels_ + threads.y - 1) / threads.y);


	std::cout << "weights_trans_: " << kernel_dim_ << " * " << out_channels_ << std::endl;
	std::cout << "output: " << out_channels_ << " * " << output_spatial_dim_ << std::endl;
	std::cout << "col_buff: " << kernel_dim_ << " * " << output_spatial_dim_ << std::endl;
	kernel_MatMul_conv<<<grid, threads>>>(weights_trans_->get_device_ptr().get(), output, col_buff, kernel_dim_, out_channels_, out_channels_, output_spatial_dim_,
			kernel_dim_, output_spatial_dim_, false);

	// col_buffer_->print_tensor("col_buffer_ after", true, 1, 9);

	// col2im
	conv_col2im_wraper(col_buff, input);
}

void Conv2D::backward_bias_gemv(float* bias, float* input)
{
	if (d_one_vec_ != nullptr) cudaFree(d_one_vec_);

	cudaMalloc((void**)&d_one_vec_, sizeof(float) * output_spatial_dim_);
	kernel_init_one_vec_conv<<< (output_spatial_dim_+BLOCK_DIM_1D-1)/BLOCK_DIM_1D, BLOCK_DIM_1D >>>(d_one_vec_, output_spatial_dim_);

	dim3 threads,grid;

	// db = (dy) * d_one_vec - dim: (out_channels_ * output_spatial_dim_) (output_spatial_dim_ * 1)
	int max_thredas= 16 * 16;
	threads = dim3(16,16); // 16 can change
	grid = dim3(1, (output_spatial_dim_ + max_thredas - 1)  / max_thredas );

	kernel_MatVec_conv<<< grid, threads>>>(input, d_one_vec_, out_channels_ , output_spatial_dim_,
				  bias, true);
}

