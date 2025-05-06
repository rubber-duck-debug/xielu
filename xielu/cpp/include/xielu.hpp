#ifndef XIELU_AUTOGRAD_HPP
#define XIELU_AUTOGRAD_HPP

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

class XIELU : public torch::CustomClassHolder {

public:
  XIELU() {}

  Tensor forward(Tensor x, Tensor alpha_p, Tensor alpha_n, double beta, double eps, bool with_vector_loads);

  std::vector<Tensor> __getstate__() { return {}; }

  void __setstate__(const std::vector<Tensor> &state) { return; }
};

#endif