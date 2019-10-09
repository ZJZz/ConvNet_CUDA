#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace std;

double poolmax_id[5][4]; // not sure about size


// Test
int channel[5] = {1,1,1,1,1};
int height[5]  = {2,2,2,2,2};
int width[5]   = {2,2,2,2,2};

double in_val[16] = {1.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0};


/*

poolsize: 2*2 strid: 2

1  1  2  4
5  6  7  8                    6  8
3  2  1  0    ---MaxPool--->  3  4
1  2  3  4


poolmax-id: 5 7 8 11

*/

double out_res[4];


void MaxPool(int cur_layer, double *input)
{
    /*-------------prior layer info-------------------*/
    int prior_rows, prior_cols, prior_channels;
    double *in_ptr;


    // Test
    prior_rows = 4;
    prior_cols = 4;
    prior_channels = 1; // same with cur layer, maybe not used in this function

    in_ptr = in_val;

    // if(cur_layer == 0)
    // {
    //     prior_rows = IMAGE_rows;
    //     prior_cols = IMAGE_cols;
    //     prior_channels = IMAGE_channels;
    //     in_ptr = input;
    // }
    // else
    // {
    //     prior_rows = height[cur_layer-1];
    //     prior_cols = width[cur_layer-1];
    //     prior_channels = channel[cur_layer-1];
    //     in_ptr = activate_val[cur_layer-1];
    // }



    /*-------------prior layer info-------------------*/


    /*-------------current layer info-------------------*/
    // pooling attributes
    int pool_size_H, pool_size_V;
    int stride_size_H, stride_size_V;

    // Test
    pool_size_H = 2;
    pool_size_V = 2;
    stride_size_H = 2;
    stride_size_V = 2;

    // pool_size_H = PoolSizeH[cur_layer];
    // pool_size_V = PoolSizeV[cur_layer];

    // stride_size_H = StrideH[cur_layer];
    // stride_size_V = StrideV[cur_layer];
    /*-------------current layer info-------------------*/


    /*-------------save output value-------------------*/
    int out_idx = 0;
    double *out_ptr;
    out_ptr = out_res;
    /*-------------save output value-------------------*/


    /*-------------compute-------------------*/
    int row_start, row_stop, col_start, col_stop;
    double value;


    for(int out_c = 0; out_c < channel[cur_layer]; out_c++)
    {
        for(int out_h = 0; out_h < height[cur_layer]; out_h++)
        {
            for(int out_w = 0; out_w < width[cur_layer]; out_w++)
            {
                // Compute activation of current layer's neuron at (out_c, out_h, out_w)
                // Pooling layer don't use padding, so don't need to worry about outbound of prior layer
                row_start = stride_size_V * out_h;
                row_stop  = row_start + pool_size_V - 1;
                col_start = stride_size_H * out_w;
                col_stop  = col_start + pool_size_H - 1;

                value = (numeric_limits<double>::min)();

                for(int in_row = row_start; in_row <= row_stop; in_row++)
                {
                    for(int in_col = col_start; in_col <= col_stop; in_col++)
                    {
                        double x = in_ptr[(out_c * prior_rows + in_row) * prior_cols + in_col];
                        if(x > value)
                        {
                            value = x;
                            poolmax_id[cur_layer][out_idx] = in_row * prior_cols + in_col; // Save max value postion, used for backprop
                        }
                    } // end in_col
                } // end in_row

                out_ptr[out_idx++] = value;
            }

        }
    }
    /*-------------compute-------------------*/
}

int main()
{

    double * org_in = nullptr;
    MaxPool(2, org_in);

    for(int i = 1; i <= 4; i++)
    {
        cout << out_res[i-1] << " ";
        if(i % 2 == 0)
            cout << endl;
    }

    for(int i = 0; i < 4; i++)
        cout << poolmax_id[2][i] << " ";
    cout << endl;


    return 0;
}
