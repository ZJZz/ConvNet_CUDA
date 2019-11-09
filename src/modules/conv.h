#ifndef _CONV_H_
#define _CONV_H_

#include "layer.h"

class Conv2D: public NNLayer
{
public:
    Conv2D(std::string name,
           int out_channels,
           int kernel_size,
           int stride=1,
           int padding=0);
    ~Conv2D();

    Tensor* forward(Tensor* input);
    Tensor* backward(Tensor* grad_output);

private:
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;

    // 权重的偏移量，尤其适用于卷积组大于1的情况
    //int weight_offset_;

    // 表示一个输出通道对应的所有卷积核对输入的一个卷积组的所有通道卷积操作一次处理数据量大小
    int kernel_dim_ = 1;



    Shape output_size_;
    int output_spatial_dim_ = 1;



    // convolution

    Tensor* col_buffer_ = nullptr;
    Tensor* col_buffer_trans_ = nullptr;
    Tensor* weights_trans_ = nullptr;

    float *d_one_vec_ = nullptr;

    void set_workspace();

    void compute_output_shape();

    void forward_gemm(float* input, float* weights, float* output);
    void forward_bias(float* output, float* bias);

    void conv_im2col_wraper(float* data, float* col_buff);
    void conv_col2im_wraper(float* col_buff, float* data);

    void conv_im2col(float* data_im, int channels,
        		         int height, int width,
        		         int kernel_h, int kernel_w,
        		         int pad_h, int pad_w,
        		         int stride_h, int stride_w,
        		         float* data_col);

    void conv_col2im(float* data_col, int channels, int height, int width,
    		         int kernel_h, int kernel_w, int pad_h, int pad_w,
    		         int stride_h, int stride_w, float* data_im);

    void backward_weight_gemm(float* input, float* ouput, float* weights);
    void backward_input_gemm(float* output, float* weights, float* input);
    void backward_bias_gemv(float* bias, float* input);

};

#endif // _CONV_H_
