#include <cstdio>
#include <iostream>

using namespace std;


/*


(ilayer)                     (ilayer + 1) this_delta

0 0 0 0 0 0 0                  

0 x x x x x 0                 1 2 3

0 x x x x x 0      kernel     4 5 6

0 x x x x x 0      1 2 3      7 8 9

0 x x x x x 0      4 5 6

0 x x x x x 0      7 8 9

0 0 0 0 0 0 0


result

5  14  10  24  15 
16 40  26  60  36 
20 44  25  54  30 
46 100 56  120 66 
35 74  40  84  45
*/

void backprop_delta_from_conv(int ilayer, double *prior_delta, double *weight_ptr, double *this_delta)
{
    /* Backprop Delta Equation
        delta_i_cur = f'(act_i_cur) * sigma( w_k_i_next * delta_k_next )
    */

    //int cur_neuron_cnt = nhid[ilayer];

    // Test
    int cur_neuron_cnt = 25;

    for(int i = 0; i < cur_neuron_cnt; i++)
        prior_delta[i] = 0.0; // zero all deltas before beginning
    
    /*-----------filter info----------*/
    // note all info is from (ilayer + 1)
    // int hwH           = HalfWidH[ilayer+1];

    // Test
    int hwH = 1;
    int kernel_width  = 2 * hwH + 1;

    // int hwV           = HalfWidV[ilayer+1];
    
    // Test
    int hwV = 1;
    int kernel_height = 2 * hwV + 1;

    // int strH = strideH[ilayer+1];
    // int strV = strideV[ilayer+1];

    // Test
    int strH = 2, strV = 2;
    
    // int pdH  = padH[ilayer+1];
    // int pdV  = padV[ilayer+1];

    // Test
    int pdH = 1, pdV = 1;
    /*-----------filter info----------*/
    

    /*-----------current layer dimension----------*/
    // int cur_height  = height[ilayer];
    // int cur_width   = width[ilayer];
    // int cur_channel = depth[ilayer];

    // Test  
    int cur_height = 5, cur_width = 5, cur_channel = 1;
    /*-----------current layer dimension----------*/

    /*-----------next layer dimension-------------*/
    // int next_height  = height[ilayer+1];
    // int next_width   = width[ilayer+1];
    // int next_channel = depth[ilayer+1];

    // Test
    int next_height = 3, next_width = 3, next_channel = 1;
    /*-----------next layer dimension-------------*/
    

    /*
    loop through every *possible* connection from a neuron in current layer to a neuron in the next layer.
    In the equation, direction is based on current layer to find connection in next layer.
    But here, we do in a reverse direction, we pick every neuron in the next layer, and loop current layer's rectangle has contribute to the next layer's delta  
    */

   int idx_next = 0;
   for(int next_slice = 0; next_slice < next_channel; next_slice++)
   {
       for(int next_row = 0; next_row < next_height; next_row++)
       {
           for(int next_col = 0; next_col < next_width; next_col++)
           {
                // find weight connect current neuron to the next neuron
                double *w_ptr = nullptr;
                // if(layer_type[ilayer + 1] == TYPE_CONV)
                //     w_ptr = layer_weights[ilayer+1] + next_slice * n_prior_weights[ilayer+1];

                // Test
                w_ptr = weight_ptr;

                // figure out prior layer rectangle
                int rstart = strV * next_row - pdV;
                int rstop  = rstart + 2 * hwV;
                int cstart = strH * next_col - pdH;
                int cstop  = cstart + 2 * hwH;

                for(int cur_slice = 0; cur_slice < cur_channel; cur_slice++)
                {
                    for(int cur_row = rstart; cur_row <= rstop; cur_row++)
                    {
                        for(int cur_col = cstart; cur_col <= cstop; cur_col++)
                        {
                            if(cur_row >= 0 && cur_row < cur_height && cur_col >=0 && cur_col < cur_width)
                            {
                                // crucial in this reverse loop to ensure correct
                                int cur_idx = (cur_slice * cur_height + cur_row) * cur_width + cur_col; 
                                prior_delta[cur_idx] += this_delta[idx_next] * *w_ptr++; // cumulate delta
                            }
                            else
                            {
                                ++w_ptr;
                            }
                        }
                    }
                }

                ++idx_next;
           }
       }
   }


    
    /*---*/
    /*---*/
    /*---*/
    /*---*/
    /*---*/

}

int main()
{

    double prior_d[25];
    double weight_ptr[9] = {1,2,3,4,5,6,7,8,9};
    double this_d[9] = {1,2,3,4,5,6,7,8,9};

    backprop_delta_from_conv(3, prior_d, weight_ptr, this_d);

    for(int i = 0; i < 25; i++)
    {
        cout << prior_d[i] << " ";
        if(i + 1 % 5 == 0)
            cout << endl;
    }



    return 0;
}
