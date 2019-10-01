#include <iostream>
#include <algorithm>
#include <cmath>
#include <limits>

using namespace std;

// Test
int channel[5] = {1,1,1,1,1};
int height[5]  = {3,3,3,3,3};
int width[5]   = {3,3,3,3,3};

double in_val[25] = {0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 1.0, 2.0, 2.0};

//double filter[9] = {0.0, 0.0, -1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 0.0}; // wrong layout

double filter[10] = {0.0, 0.0, -1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 0.0, 1.0}; // include bias

double res[9];

/*

filter size: 3*3 strid: 2 pad: 1

    -1  0  1  2  3  4  5
 -1  0  0  0  0  0  0  0
  0  0  0  2  0  0  2  0        0  0 -1                1  -4  -3
  1  0  1  2  2  0  1  0   *   -1 -1  1      ------>  -5  -1  -2 (no bias, no activation)
  2  0  1  0  0  0  2  0        0 -1  0                0  -3  -4
  3  0  2  0  1  2  0  0
  4  0  2  2  1  2  2  0
  5  0  0  0  0  0  0  0
*/



void Conv2d(int cur_layer, double *input)
{
    /*-------------prior layer info-------------------*/

    int prior_rows, prior_cols, prior_channels;
    double *in_ptr;

    prior_rows = 5;
    prior_cols = 5;
    prior_channels = 1;
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

    int halfwid_H = 1;
    int halfwid_V = 1;

    int pad_size_H = 1;
    int pad_size_V = 1;

    int stride_size_H = 2;
    int stride_size_V = 2;
    // // filter size
    // int halfwid_H = HalfWidH[cur_layer];
    // int halfwid_V = HalfWidV[cur_layer];

    // // pad size
    // int pad_size_H = PadH[cur_layer];
    // int pad_size_V = PadV[cur_layer];

    // // stride size
    // int stride_size_H = StrideH[cur_layer];
    // int stride_size_V = StrideV[cur_layer];

    /*-------------current layer info-------------------*/


    /*-------------save output value-------------------*/
    int out_idx = 0;
    double *out_ptr = res;

    // out_ptr = activity[cur_layer];
    /*-------------save output value-------------------*/


    /*-------------compute-------------------*/
    for(int out_c = 0; out_c < channel[cur_layer]; out_c++)
    {
        // The weights for current layer are same for all neurons in the layer's visual field,but a different such set used for each slice
        // double *wt_ptr = layer_weights[cur_layer] + out_c * n_prior_weights[cur_layer];



        for(int out_h = 0; out_h < height[cur_layer]; out_h++)
        {
            for(int out_w = 0; out_w < width[cur_layer]; out_w++)
            {
                double *wt_ptr = filter;

                // Compute activation of current layer's neuron at (out_c, out_h, out_w)
                double sum = 0.0;

                // center of first filter is at HalfWidth-Pad
                // filter begins at -Pad
                int row_start = stride_size_V * out_h - pad_size_V;
                int row_stop  = row_start + 2 * halfwid_V;
                int col_start = stride_size_H * out_w - pad_size_H;
                int col_stop  = col_start + 2 * halfwid_H;

                cout << "row_start: " << row_start << " row_stop: " << row_stop << endl;
                cout << "col_start: " << col_start << " col_stop: " << col_stop << endl;

                for(int in_slice = 0; in_slice < prior_channels; in_slice++)
                {
                    for(int in_row = row_start; in_row <= row_stop; in_row++)
                    {
                        for(int in_col = col_start; in_col <= col_stop; in_col++)
                        {
                            double x;
                            // boundary check
                            if(in_row >= 0 && in_row < prior_rows && in_col >=0 && in_col < prior_cols)
                                x = in_ptr[ (in_slice * prior_rows + in_row) * prior_cols + in_col ];
                            else
                                x = 0.0;

                            cout << x << " * " << *wt_ptr << "   ";

                            sum += x * *wt_ptr++;

                        } // end for in_col
                        cout << endl;
                    } // end for in_row
                } // end for in_slice


                // add bias
                sum += *wt_ptr++;
                // activation function
//                sum =  exp(2.0 * sum);
//                sum = (sum -1.0) / (sum + 1.0);
                
//                cout << "output posistion res: (" << out_h << "," << out_w << ") -> " << sum << endl;

                // save result for this neuron
                out_ptr[out_idx++] = sum;
            }
        }
    }
    /*-------------compute-------------------*/
}

int main()
{

    double *org_input = nullptr;
    Conv2d(4, org_input);

    for(int i = 1; i <= 9; i++)
    {
        cout << res[i-1] << "  ";
        if(i % 3 == 0)
            cout << endl;
    }

    return 0;
}
