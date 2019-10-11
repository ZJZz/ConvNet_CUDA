#include <iostream>
#include <cmath>

using namespace std;

// 3 classes
double output[3] = {0,0,0}; // one case in a batch

double labels[3] = {0.0 , 1.0, 0.0};

double loss;

void softmax()
{
    double sum = 1.e-60;
    for(int i = 0; i < 3; i++)
    {
        if(output[i] < 300.0)
        {
            output[i] = exp(output[i]);
        }
        else
        {
            output[i] = exp(300.0);
        }
        sum += output[i];
    }

    for(int i = 0; i < 3; i++)
        output[i] /= sum;
}

void compute_loss(int batch_start, int batch_end)
{
    loss = 0.0;

    double err;

    int imax;
    for(int icase = batch_start; icase < batch_end; icase++)
    {
        err = 0.0;

        double tmax = -1.e30;
        imax = 0;

        for(int i = 0; i < 3; i++)
        {
            if(labels[i] > tmax)
            {
                imax = i;
                tmax = labels[i];
            }
        }
        err   = -log(output[imax] + 1.e-30);
        loss += err;
    }

}


int main()
{
    softmax();
    // for(int i = 0; i  < 3; i++)
    //     cout << output[i] << " ";
    // cout << endl;
    compute_loss(0,1);
    cout << loss << endl; // 1.09861
    return 0;
}
