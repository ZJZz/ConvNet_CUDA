#include <cstdio>
#include <iostream>


/*
poolsize: 2*2 strid: 2

1  1  2  4
5  6  7  8                    6  8
3  2  1  0    ---MaxPool--->  3  4
1  2  3  4

poolmax-id: 5 7 8 11
*/

int poolmax_id[4] = {5, 7, 8, 11};

double prior_delta[16];

double this_delta[4] = {1,1,1,1};

void backprop_delta_from_maxpool(int ilayer)
{
    /* Backprop Delta Equation
        delta_i_cur = f'(act_i_cur) * sigma( w_k_i_next * delta_k_next )
    */

    // int cur_neuron_cnt = nhid[ilayer];
    // Test
    int cur_neuron_cnt = 16;

    for(int i = 0; i < cur_neuron_cnt; i++)
        prior_delta[i] = 0.0; // about to compute
    
    /*-----------pool info----------*/
    // note all infor is from (ilayer + 1)
    // int poolH = PoolWidH[ilayer+1];
    // int poolV = PoolWidV[ilayer+1];
    // int strH = strideH[ilayer+1];
    // int strV = strideV[ilayer+1];

    // Test
    int poolH = 2;
    int poolV = 2;
    int strH = 2;
    int strV = 2;
    /*-----------pool info----------*/
    

    /*-----------current layer dimension----------*/
    // int cur_height  = height[ilayer];
    // int cur_width   = width[ilayer];
    // int cur_channel = depth[ilayer];

    // Test
    int cur_height  = 4;
    int cur_width   = 4;
    int cur_channel = 1;  
    /*-----------current layer dimension----------*/

    /*-----------next layer dimension-------------*/
    // int next_height  = height[ilayer+1];
    // int next_width   = width[ilayer+1];
    // int next_channel = depth[ilayer+1];

    // Test
    int next_height  = 2;
    int next_width   = 2;
    int next_channel = 1;
    /*-----------next layer dimension-------------*/
    
   int idx_next = 0;
   for(int next_slice = 0; next_slice < next_channel; next_slice++)
   {
       for(int next_row = 0; next_row < next_height; next_row++)
       {
           for(int next_col = 0; next_col < next_width; next_col++)
           {
                // int cur_row = poolmax_id[ilayer+1][idx_next] / cur_width;
                // int cur_col = poolmax_id[ilayer+1][idx_next] % cur_width;
                int cur_row = poolmax_id[idx_next] / cur_width;
                int cur_col = poolmax_id[idx_next] % cur_width;

                int idx_cur = (next_slice * cur_height + cur_row) * cur_width + cur_col;
                prior_delta[idx_cur] += this_delta[idx_next];

                ++idx_next;
           }
       }
   }
}

int main()
{

    backprop_delta_from_maxpool(5);

    for(int i = 0; i < 16; i++)
    {
        std::cout << prior_delta[i] << " ";
        if((i + 1) % 4 == 0)
            std::cout << std::endl;
    }


    return 0;
}