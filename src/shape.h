#ifndef _SHAPE_H_
#define _SHAPE_H_


struct Shape {

    Shape(int n=1, int c=1, int h=1, int w=1);

    int n() const { return n_; }
    int c() const { return c_; }
    int h() const { return h_; }
    int w() const { return w_; }

    // return number of total elements for 1 picture
    int size_per_batch() { return c_ * h_ * w_; }

    // return number of total elements in Tensor including batch
    int total_elements() { return n_ * c_ * h_ * w_; }

    /* data */
    int n_,c_,h_,w_;
};

#endif // _SHAPE_H_
