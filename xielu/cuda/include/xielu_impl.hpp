#ifndef XIELU_IMPL_HPP
#define XIELU_IMPL_HPP

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

class XIELUAutograd : public Function<XIELUAutograd> {
public:
  static torch::Tensor forward(AutogradContext *ctx, Tensor x, Tensor alpha_p, Tensor alpha_n, double beta, double eps, bool with_vector_loads);

  static variable_list backward(AutogradContext *ctx,
                                variable_list grad_outputs);
};

#endif
