#include <iostream>
#include <cstdio>


using namespace std;

double grad_pt[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

double wt_ptr[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
double prior_delta[3] = {7.0, 8.0, 9.0};

double activation[3] = {4.0, 5.0, 6.0};
double activation_prior[3] = {1.0, 2.0, 3.0};


/*
result:

28 56 84 
40 80 120 
54 108 162
*/

void grad_fc_layer(int icase, int ilayer)
{

    /*
    Equation:

        dLoss/dw_i_j_ilayer = (act_j_ilayer-1) * (delta_i_ilayer)
    */

    // int neuron_cur = nhid[ilayer];

    // Test
    int neuron_cur = 3;

    // get next layer neuron cnt
    int neuron_next;
    // if(ilayer == n_layers -1)
    //     neuron_next = n_classes;
    // else
    // {
    //     neuron_next = nhid[ilayer+1];
    // }

    // Test
    neuron_next = 3;


    // get prior layer activation
    // if(ilayer == 0)
    // {
    //     pre_act = database + icase * n_db_cols;
    // }
    // else
    // {
    //     pre_act = activation[ilayer-1];
    // }

    // Test
    double *pre_act = activation_prior;

    // double *grad_ptr   = layer_gradient[ilayer];
    // double *weight_ptr = layer_weights[ilayer+1]; // connect (ilayer-1) neuron to (ilayer) neuron

    double *grad_ptr   = grad_pt;
    double *weight_ptr = wt_ptr; // connect (ilayer-1) neuron to (ilayer) neuron



    for(int i = 0; i < neuron_cur; i++)
    {
        cout << "i: " << i << endl;
        double delta = 0.0;
        // if(ilayer + 1 == n_layers || layer_type[ilayer+1] == TYPE_FC)
        // {

        //     for(int j = 0; j < neuron_next; j++)
        //         delta += this_delta[j] * weight_ptr[j * (neuron_cur + 1) + i];
        // }
        // else if(i == 0)
        // {
        //     if(layer_type[ilayer+1] == TYPE_CONV)
        //         compute_nonpooled_delta(ilayer);
        //     else if(layer_type[ilayer+1] == TYPE_POOLMAX)
        //         compute_pooled_delta(ilayer);
        //     delta = prior_delta[i];
        // }
        // else
        // {
        //     delta = prior_delta[i];
        // }

        // Test
        delta = prior_delta[i];

        // delta *= 1.0 - activation[ilayer][i] * activation[ilayer][i];

        // Test
        delta *= activation[i];

        // prior_delta[i] = delta;


        cout << "------" << endl;

        for(int j = 0; j < 3; j++)
        {

            cout << "j: " << j << endl;
            cout << delta << " * " << pre_act[j] << endl;
            *grad_ptr += delta * pre_act[j];
//            cout << "finished compute" << endl;
            grad_ptr++;


        }


        // for(int j = 0; j < n_prior_weights[ilayer]-1; j++)
        //     *grad_ptr++ += delta * pre_act[j];
        // *grad_ptr++ += delta; // bias
        cout << endl;
    }
}

int main()
{
    grad_fc_layer(5, 3);

    cout << "end grad_fc_layer" << endl;

    for(int i = 0; i < 9; i++)
    {
        cout << grad_pt[i] << " ";
        if((i + 1) % 3 == 0)
            cout << endl;
    }
    return 0;
}
