#include "linear.h"


#define BLOCK_DIM_1D    512
#define BLOCK_DIM       16

// Thread block size
#define TILE_DIM 16


__global__ void kernel_transpose(float *odata, float *idata, int width, int height)
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

__global__ void kernel_MatMul(float* A, float* B, float* C, int ARows, int ACols, int BRows,
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

__global__ void kernel_MatVec(float *device_Mat, float *device_Vect,int matRowSize, int vlength, float *device_ResVect)
{
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int tindex= tidx + gridDim.x * BLOCK_DIM * tidy;


	if(tindex<matRowSize)
	{
		int i;int m= tindex * vlength;
		device_ResVect[tindex]=0.0f;
		for(i = 0; i < vlength; i++)
			device_ResVect[tindex] +=  device_Mat[m+i] * device_Vect[i];
	}

	__syncthreads();

}



__global__ void init_one_vec(float* d_one_vec, size_t length)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= length) return;

	d_one_vec[i] = 1.f;
}

Linear::Linear(std::string name, int output_size)
{
	name_ = name;
	output_size_ = output_size;
}

Linear::~Linear()
{
	if (d_one_vec != nullptr)
		cudaFree(d_one_vec);

	if (weights_trans_ != nullptr)
		delete weights_trans_;
}


Tensor* Linear::forward(Tensor* input)
{
	// initialize weights and biases
	if (weights_ == nullptr)
	{
		// setup parameter size information
		input_size_  = input->c() * input->h() * input->w();

		// initialize weight, bias, and output
		weights_ = new Tensor(1, 1, input_size_, output_size_);
		weights_trans_ = new Tensor(1, 1, output_size_, input_size_);
		biases_  = new Tensor(1, 1, output_size_);

	}

	// initilaize input and output
	if (input_ == nullptr || batch_size_ != input->n())
	{
		input_       = input;
		batch_size_  = input->n();

		if (output_ == nullptr)
			output_  = new Tensor(batch_size_, output_size_);
		else
		{
			//output_->reset(batch_size_, output_size_);
			std::cout << "Error: Linear::forward - output_ already exist, need reshape" << std::endl;
		}

		if (d_one_vec != nullptr) cudaFree(d_one_vec);
		cudaMalloc((void**)&d_one_vec, sizeof(float) * batch_size_);
		init_one_vec<<< (batch_size_+BLOCK_DIM_1D-1)/BLOCK_DIM_1D, BLOCK_DIM_1D >>>(d_one_vec, batch_size_);

		// initialize weights and biases
		if (!freeze_)
		{
			init_weight_bias(0);
		}
	}

	dim3 threads,grid;

	std::cout << "before transpose" << std::endl;

	// transpose weights
	threads = dim3(TILE_DIM, TILE_DIM);
	grid = dim3((output_size_ + threads.x - 1) / threads.x, (input_size_ + threads.y - 1) / threads.y );
	kernel_transpose<<<grid, threads>>>(weights_trans_->get_device_ptr().get(),
			          weights_->get_device_ptr().get(),
			          output_size_, input_size_);

	std::cout << "after transpose" << std::endl;

	// weight MatMul
	threads = dim3(TILE_DIM, TILE_DIM);
	grid = dim3((batch_size_ + threads.x - 1) / threads.x, (output_size_ + threads.y - 1) / threads.y);

	kernel_MatMul<<<grid, threads>>>(weights_trans_->get_device_ptr().get(),
			       input_->get_device_ptr().get(),
			       output_->get_device_ptr().get(),
			       output_size_,  input_size_,
			       input_size_, batch_size_,
			       output_size_, batch_size_,false);

	 // bias
	kernel_MatMul<<<grid, threads>>>(biases_->get_device_ptr().get(),
			 	 	 	  d_one_vec,
	 			       output_->get_device_ptr().get(),
	 			       output_size_,  1,
	 			       1, batch_size_,
	 			       output_size_, batch_size_,true);


	std::cout << "finished" << std::endl;

	 return output_;
}

Tensor* Linear::backward(Tensor* grad_output)
{
	std::cout << "In " << name_ << " backward" << std::endl;
	if (grad_weights_ == nullptr)
	{
		grad_weights_ = new Tensor(weights_->shape());
		grad_biases_  = new Tensor(biases_->shape());
	}

	if (grad_input_ == nullptr || batch_size_ != grad_output->n())
	{
		grad_output_  = grad_output;
		grad_output_trans_ = new Tensor(1,1,batch_size_, output_size_);

		if (grad_input_ == nullptr)
			grad_input_   = new Tensor(input_->shape());
		else
		{
			std::cout << "Error: Linear::backward - grad_input_ already exist, need reshape" << std::endl;
			//grad_input_->reset(input_->shape());
		}
	}

	dim3 threads,grid;

	// db = (dy) * d_one_vec - dim: (output_size * batch_size) (batch_size * 1)
	int max_thredas= 16 * 16;
	threads = dim3(16,16); // 16 can change
	grid = dim3(1, (batch_size_ + max_thredas - 1)  / max_thredas );

	kernel_MatVec<<< grid, threads>>>(grad_output_->get_device_ptr().get(), d_one_vec, output_size_, batch_size_,
			      grad_biases_->get_device_ptr().get());


	// (dy)^T
	threads = dim3(TILE_DIM, TILE_DIM);
	grid = dim3((batch_size_ + threads.x - 1) / threads.x, (output_size_ + threads.y - 1) / threads.y);
	kernel_transpose<<<grid,threads >>>(grad_output_trans_->get_device_ptr().get(),
			         grad_output_->get_device_ptr().get(),
			         batch_size_, output_size_);


	grid = dim3((output_size_ + threads.x - 1) / threads.x, (input_size_ + threads.y - 1) / threads.y);
	// dw = x * (dy)^T - dim: (input_size_ * batch_size) (batch_size * output_size_)
	kernel_MatMul<<< grid, threads>>>(input_->get_device_ptr().get(), grad_output_trans_->get_device_ptr().get(),
			               grad_weights_->get_device_ptr().get(),
			               input_size_, batch_size_,
			               batch_size_, output_size_,
			               input_size_, output_size_, false);

	if(!gradient_stop_)
	{
		// dx = W * dy - dim: (input_size_, output_size_) (output_size * batch_size)
		threads = dim3(TILE_DIM, TILE_DIM);
		grid = dim3((batch_size_ + threads.x - 1) / threads.x, (input_size_ + threads.y - 1) / threads.y);
		kernel_MatMul<<<grid, threads >>>(weights_->get_device_ptr().get(),
				               grad_output_->get_device_ptr().get(),
				               grad_input_->get_device_ptr().get(),
				               input_size_, output_size_,
				               output_size_, batch_size_,
				               input_size_, batch_size_, false);
	}

	return grad_input_;

}
