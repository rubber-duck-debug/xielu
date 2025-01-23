#include "utils.cuh"
#include "xielu_impl.hpp"

#include <iostream>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;

#define CHECK_RESULT(result)                                                   \
  if (result != cudaSuccess) {                                                 \
    throw runtime_error(string("Encountered error ") +                         \
                        cudaGetErrorName(result) + " at " + __FILE__ + ": " +  \
                        to_string(__LINE__));                                  \
  }

static int getMaxBlocks() {
  // Get an upper limit on how many thread blocks we try to launch based on the
  // size of the GPU.
  int device, numMultiprocessors;
  CHECK_RESULT(cudaGetDevice(&device));
  CHECK_RESULT(cudaDeviceGetAttribute(&numMultiprocessors,
                                      cudaDevAttrMultiProcessorCount, device));
  return numMultiprocessors * 4;
}

template <typename scalar_t>
__global__ void forward_kernel(const Accessor<scalar_t, 3> x,
                               const Accessor<scalar_t, 1> alpha_p,
                               const Accessor<scalar_t, 1> alpha_n,
                               const scalar_t beta, const scalar_t eps,
                               Accessor<scalar_t, 3> output) {

  const int batch_size = x.size(0);
  const int seq_len = x.size(1);
  const int hidden_dim = x.size(2);
}

template <typename scalar_t>
__global__ void backward_kernel(const Accessor<scalar_t, 3> x,
                                const Accessor<scalar_t, 1> alpha_p,
                                const Accessor<scalar_t, 1> alpha_n,
                                const Accessor<scalar_t, 3> grad_outputs,
                                const scalar_t beta, const scalar_t eps,
                                Accessor<scalar_t, 3> dx,
                                Accessor<scalar_t, 1> dalpha_p,
                                Accessor<scalar_t, 1> dalpha_n) {

  const int batch_size = x.size(0);
  const int seq_len = x.size(1);
  const int hidden_dim = x.size(2);
}

torch::Tensor XIELUAutograd::forward(AutogradContext *ctx, Tensor x,
                                     Tensor alpha_p, Tensor alpha_n,
                                     double beta, double eps) {

  PUSH_RANGE("XIELU_FWD", 0)

  const int batch_size = x.size(0);
  const int seq_len = x.size(1);
  const int hidden_dim = x.size(2);

  const int blockSize = 128;
  const int numBlocks =
      max(1, min(getMaxBlocks(),
                 ((batch_size * seq_len * hidden_dim) + blockSize - 1) /
                     blockSize));

  TensorOptions options = x.options();
  Tensor output = torch::empty_like(x);

  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const c10::cuda::CUDAStreamGuard guard(stream);

  AT_DISPATCH_FLOATING_TYPES(
      x.scalar_type(), "forward", ([&] {
        // size_t space = 0;
        // void *sptr;
        forward_kernel<<<numBlocks, blockSize, 0, stream>>>(
            get_accessor<scalar_t, 3>(x), get_accessor<scalar_t, 1>(alpha_p),
            get_accessor<scalar_t, 1>(alpha_n), (scalar_t)beta, (scalar_t)eps,
            get_accessor<scalar_t, 3>(output));
      }));

  ctx->save_for_backward({x, alpha_p, alpha_n});
  ctx->saved_data["eps"] = eps;
  ctx->saved_data["beta"] = beta;

  POP_RANGE

  return output;
}

variable_list XIELUAutograd::backward(AutogradContext *ctx,
                                      variable_list grad_outputs) {

  PUSH_RANGE("XIELU_BWD", 1)

  auto saved = ctx->get_saved_variables();
  Tensor x = saved[0];
  Tensor alpha_p = saved[1];
  Tensor alpha_n = saved[2];
  const double eps = ctx->saved_data["eps"].toDouble();
  const double beta = ctx->saved_data["beta"].toDouble();

  const int nbatch = x.size(0);
  const int seq_len = x.size(1);
  const int hidden_dim = x.size(2);

  const int blockSize = 128;
  const int numBlocks = max(
      1, min(getMaxBlocks(),
             ((nbatch * seq_len * hidden_dim) + blockSize - 1) / blockSize));

  TensorOptions options = x.options();
  Tensor dx = torch::empty_like(x);
  // each block stores a contribution to dalpha_p, dalpha_n
  Tensor dalpha_p = torch::empty({numBlocks}, options);
  Tensor dalpha_n = torch::empty({numBlocks}, options);

  /*might not be needed - can check performance with/without.
    (contiguity isn't guaranteed for grad_outputs!)
  */
  grad_outputs[0] = grad_outputs[0].contiguous();

  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const c10::cuda::CUDAStreamGuard guard(stream);

  AT_DISPATCH_FLOATING_TYPES(
      x.scalar_type(), "backward", ([&] {
        // size_t space = 0;
        // void *sptr;
        backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
            get_accessor<scalar_t, 3>(x), get_accessor<scalar_t, 1>(alpha_p),
            get_accessor<scalar_t, 1>(alpha_n),
            get_accessor<scalar_t, 3>(grad_outputs[0]), (scalar_t)beta,
            (scalar_t)eps, get_accessor<scalar_t, 3>(dx),
            get_accessor<scalar_t, 1>(dalpha_p),
            get_accessor<scalar_t, 1>(dalpha_n));
      }));

  POP_RANGE

  return {dx, torch::sum(dalpha_p), torch::sum(dalpha_n)};
}