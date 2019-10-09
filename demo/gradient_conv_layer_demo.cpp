#include <iostream>
#include <cstdio>

using namespace std;

float activation_cur[9] = {1.0, -4.0, -3.0, -5.0, -1.0, -2.0, 0.0, -3.0, -4.0};
float activation_pre[25] = {0.0, 2.0, 0.0, 0.0, 2.0, 1.0, 2.0, 2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 0.0, 2.0, 2.0, 1.0, 2.0, 2.0};
float kernel_weight[9] = {0.0, 0.0, -1.0, -1.0, -1.0, 1.0, 0.0, -1.0, 0.0};
float host_delta[9] = {1,2,3,4,5,6,7,8,9};
float kernel_grad[9];

/*

0 2 0 0 2            k          output         delta

1 2 2 0 1          0 0 -1       1  -4  -3     1  2  3

1 0 0 0 2         -1 -1 1      -5  -1  -2     4  5  6

2 0 1 2 0          0 -1 0       0  -3  -4     7  8  9

2 2 1 2 2


result

298 206 344 
496 562 146 
112 297 12 

*/


void grad_conv(int icase, int ilayer)
{

    // int n_cur = nhid[ilayer];

    // Test
    int n_cur = 9;

    // int n_next;
    // if(ilayer == n_layers - 1)
    //     n_next = n_classes;
    // else
    //     n_next = nhid[ilayer+1];

    /*--------prior layer info----------*/
    // float pre_act = activation[ilayer-1];
    // int in_height = height[ilayer-1];
    // int in_width  = width[ilayer-1];
    // int in_channels = depth[ilayer-1];

    // Test
    float *pre_act = activation_pre;
    int in_height = 5;
    int in_width  = 5;
    int in_channels = 1;


    /*--------prior layer info----------*/


    float *w_ptr = kernel_weight; // connect ilayer to ilayer + 1

    int idx_cur = 0;
    for(int cur_slice = 0; cur_slice < 1; cur_slice++)
    {
        for(int cur_row = 0; cur_row < 3; cur_row++)
        {
            for(int cur_col = 0; cur_col < 3; cur_col++)
            {
                cout << "(" << cur_row << "," << cur_col << ")" << endl;
                float delta = 0.0;
                // if(ilayer + 1 == n_layers || layer_type[ilayer+1] == TYPE_FC)
                // {

                //     for(int j = 0; j < n_next; j++)
                //         delta += this_delta[j] * w_ptr[j * (n_cur + 1) + i];
                // }
                // else if(cur_slice == 0 && cur_row == 0 && cur_col == 0 )
                // {
                //     if(layer_type[ilayer+1] == TYPE_CONV)
                //         compute_nonpooled_delta(ilayer);
                //     else if(layer_type[ilayer+1] == TYPE_POOLMAX)
                //         compute_pooled_delta(ilayer);
                //     delta = prior_delta[idx_cur];
                // }
                // else
                // {
                //     delta = prior_delta[idx_cur];
                // }

                // Test
                delta = host_delta[idx_cur];

                // delta *= 1.0 - activation[ilayer][i] * activation[ilayer][i];

                delta *= activation_cur[idx_cur] * activation_cur[idx_cur];

                cout << "delta: " << delta << endl;

                // prior_delta[idx_cur] = delta;

                // reset according to the current slice every time we begin processing a new neuron in the current visual field
                float *grad_ptr = kernel_grad;

                // int row_start = strideV[ilayer] * cur_row - padV[ilayer];
                // int row_end   = row_start + 2 * HalfWidV[ilayer];
                // int col_start = strideH[ilayer] * cur_row - padH[ilayer];
                // int col_end   = col_start + 2 * HalfWidH[ilayer];

                int row_start = 2 * cur_row - 1;
                int row_end   = row_start + 2 * 1;
                int col_start = 2 * cur_col - 1;
                int col_end   = col_start + 2 * 1;


                for(int in_slice = 0; in_slice < in_channels; in_slice++)
                {
                    for(int in_row = row_start; in_row <= row_end; in_row++)
                    {
                        for(int in_col = col_start; in_col <= col_end; in_col++)
                        {

                            float x;
                            if(in_row >= 0 && in_row < in_height && in_col >=0 && in_col < in_width)
                            {
                                cout << "   (" << in_row << "," << in_col << ")" << endl;
                                x = pre_act[(in_slice * in_height + in_row) * in_width + in_col];
                            }
                            else
                                x = 0.0;
                            if((in_row - row_start == 2)  && (in_col - col_start == 1))
                                cout << delta << " * " << x << endl;
                            *grad_ptr++ += delta * x;
                        }
                    }
                }

                // *grad_ptr++ += delta;  // bias
                ++idx_cur;
            }
        }
    }


}

int main()
{

    grad_conv(5, 5);
    for(int i = 0; i < 9; i++)
    {
        cout << kernel_grad[i] << " ";
        if((i + 1) % 3 == 0)
            cout << endl;
    }

}
