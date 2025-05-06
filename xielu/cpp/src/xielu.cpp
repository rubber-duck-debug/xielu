#include "xielu.hpp"
#include "xielu_impl.hpp"

#include <iostream>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;

// wrapper class which we expose to the API.
torch::Tensor XIELU::forward(Tensor x, Tensor alpha_p, Tensor alpha_n, double beta, double eps, bool with_vector_loads) {
  return XIELUAutograd::apply(x, alpha_p, alpha_n, beta, eps, with_vector_loads);
}

TORCH_LIBRARY(xielu, m) {
  m.class_<XIELU>("XIELU")
      .def(torch::init<>(), "", {})
      .def("forward", &XIELU::forward)
      .def_pickle(
          [](const c10::intrusive_ptr<XIELU> &self)
              -> std::vector<torch::Tensor> { return self->__getstate__(); },
          [](const std::vector<torch::Tensor> &state)
              -> c10::intrusive_ptr<XIELU> {
            auto obj = c10::make_intrusive<XIELU>();
            obj->__setstate__(state);
            return obj;
          });
}