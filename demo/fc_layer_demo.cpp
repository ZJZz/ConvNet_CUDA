#include <iostream>
#include <algorithm>
#include <cmath>

using namespace std;

int n_layer = 5;
int n_classes = 10;

// double layer_weights[2][3] = {{1.0, 2.0, 3.0},{4.0, 5.0, 6.0}}; wrong layout
// double layer_weights[6] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};       wrong layout, bias not included
double layer_weights[8] = {1.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 1.0};

double val_in[3] = {1.0, 2.0, 3.0};
double val_out[2];

double org_input[4];

int in_cnt = 3;
int out_cnt = 2;


void Linear(int ilayer, const double *input, int activation)
{
    // weights for connect prior layer to this layer
    double *wt_ptr;

    wt_ptr = layer_weights;

    // info about prior layer
    int num_in_neuron;
    double *val_in_neuron;

    // Test
    num_in_neuron = in_cnt;
    val_in_neuron = val_in;


    // if(ilayer == 0) // data is input
    // {
    //     num_in_neuron = n_pred;
    //     val_in_neuron = input;
    // }
    // else // activation value is input
    // {
    //     num_in_neuron = nhid[ilayer-1];
    //     val_in_neuron = activ_val[ilayer-1];

    // }




    // info about this layer
    int num_out_neuron;
    double *val_out_neuron;

    // Test
    num_out_neuron = out_cnt;
    val_out_neuron = val_out;


    // if(ilayer == n_layer)
    // {
    //     num_out_neuron = n_classes;
    //     val_out_neuron = output;
    // }
    // else
    // {
    //     num_out_neuron = nhid[ilayer];
    //     val_out_neuron = activ_val[ilayer-1];
    // }

    double sum = 0.0;

    // compute this layer's all neuron value
    for(int i_out = 0; i_out < num_out_neuron; i_out++)
    {
        sum = 0.0;
        for(int i_in = 0; i_in < num_in_neuron; i_in++)
        {
//            cout << val_in_neuron[i_in] <<" * " << *wt_ptr << endl;
            // *wt_ptr++ = (*wt_ptr) then wt_ptr++
            sum += val_in_neuron[i_in] * (*wt_ptr++);
        }
        sum += *wt_ptr++;

        if(activation)
        {
            sum = exp(2.0 * sum);
            sum = (sum - 1.0) / (sum + 1.0);
        }

        val_out_neuron[i_out] = sum;
    }

}

int main()
{

    Linear(3, org_input, 1);

    for(int i = 0; i < 2; i++)
        cout << val_out[i] << " ";
    cout << endl;

    return 0;
}
