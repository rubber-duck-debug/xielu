#include "utils.cuh"
#include "xielu_impl.hpp"

#include <iostream>

#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;
using namespace c10;

using torch::Tensor;
using torch::TensorOptions;

#define NWARPS 4
#define WARP_SIZE 32

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

template <typename T> struct softplus {
  static __device__ T f(T x) {
    return x > T(20.0) ? x : (x < T(-20.0) ? T(0.0) : log1p(exp(x)));
  }

  static __device__ T df(T x) {
    return x > T(20.0) ? T(1.0)
                       : (x < T(-20.0) ? T(0.0) : T(1.0) / (T(1.0) + exp(-x)));
  }
};

template <> struct softplus<c10::Half> {
  static __device__ c10::Half f(c10::Half x) {
    return static_cast<c10::Half>(softplus<float>::f(static_cast<float>(x)));
  }
  static __device__ c10::Half df(c10::Half x) {
    return static_cast<c10::Half>(softplus<float>::df(static_cast<float>(x)));
  }
};

template <> struct softplus<c10::BFloat16> {
  static __device__ c10::BFloat16 f(c10::BFloat16 x) {
    return static_cast<c10::BFloat16>(
        softplus<float>::f(static_cast<float>(x)));
  }
  static __device__ c10::BFloat16 df(c10::BFloat16 x) {
    return static_cast<c10::BFloat16>(
        softplus<float>::df(static_cast<float>(x)));
  }
};

template <typename scalar_t>
__global__ void forward_kernel(const Accessor<scalar_t, 3> x,
                               const Accessor<scalar_t, 1> alpha_p,
                               const Accessor<scalar_t, 1> alpha_n,
                               const scalar_t beta, const scalar_t eps,
                               Accessor<scalar_t, 3> output) {

  const int batch_size = x.size(0);
  const int seq_len = x.size(1);
  const int hidden_dim = x.size(2);
  const int total_elements = batch_size * seq_len * hidden_dim;

  using sp = softplus<scalar_t>;
  const scalar_t s_alpha_p = sp::f(alpha_p[0]);
  const scalar_t s_alpha_n = sp::f(alpha_n[0]);

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
       idx += blockDim.x * gridDim.x) {

    int batch_idx = idx / (seq_len * hidden_dim);
    int residual = idx - batch_idx * seq_len * hidden_dim;
    int seq_idx = residual / hidden_dim;
    int hidden_idx = residual - seq_idx * hidden_dim;

    scalar_t x_e = x[batch_idx][seq_idx][hidden_idx];

    if (static_cast<float>(x_e) > 0.0f) {
      output[batch_idx][seq_idx][hidden_idx] =
          s_alpha_p * x_e * x_e + beta * x_e;
    } else {
      output[batch_idx][seq_idx][hidden_idx] =
          (beta + s_alpha_n) * expm1(min(x_e, eps)) - s_alpha_n * x_e;
    }
  }
}

template <typename scalar_t, typename reduction_type>
__global__ void backward_kernel(const Accessor<scalar_t, 3> x,
                                const Accessor<scalar_t, 1> alpha_p,
                                const Accessor<scalar_t, 1> alpha_n,
                                const Accessor<scalar_t, 3> grad_outputs,
                                const scalar_t beta, const scalar_t eps,
                                Accessor<scalar_t, 3> dx,
                                Accessor<reduction_type, 1> dalpha_p,
                                Accessor<reduction_type, 1> dalpha_n
                                ) {

  const int batch_size = x.size(0);
  const int seq_len = x.size(1);
  const int hidden_dim = x.size(2);
  const int total_elements = batch_size * seq_len * hidden_dim;

  using sp = softplus<scalar_t>;
  const scalar_t _alpha_p = alpha_p[0];
  const scalar_t _alpha_n = alpha_n[0];

  const scalar_t s_alpha_p = sp::f(_alpha_p);
  const scalar_t s_alpha_n = sp::f(_alpha_n);

  reduction_type thread_dalpha_p = 0.0;
  reduction_type thread_dalpha_n = 0.0;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total_elements;
       idx += blockDim.x * gridDim.x) {

    int batch_idx = idx / (seq_len * hidden_dim);
    int residual = idx - batch_idx * seq_len * hidden_dim;
    int seq_idx = residual / hidden_dim;
    int hidden_idx = residual - seq_idx * hidden_dim;

    const scalar_t x_e = x[batch_idx][seq_idx][hidden_idx];
    const scalar_t grad_output = grad_outputs[batch_idx][seq_idx][hidden_idx];

    if (static_cast<float>(x_e) > 0.0f) {
      dx[batch_idx][seq_idx][hidden_idx] =
          grad_output * (2 * s_alpha_p * x_e + beta);
      thread_dalpha_p += grad_output * sp::df(_alpha_p) * x_e * x_e;
    } else {
      dx[batch_idx][seq_idx][hidden_idx] =
          grad_output * ((beta + s_alpha_n) * exp(min(x_e, eps)) - s_alpha_n);
      thread_dalpha_n +=
          grad_output * sp::df(_alpha_n) * (expm1(min(x_e, eps)) - x_e);
    }
  }

  __syncthreads();

  // reduce thread-local contributions into thread % 32 = 0
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    thread_dalpha_p += __shfl_down_sync(0xFFFFFFFF, thread_dalpha_p, offset);
    thread_dalpha_n += __shfl_down_sync(0xFFFFFFFF, thread_dalpha_n, offset);
  }

  // write each warp's contributions to grad to gmem
  if (threadIdx.x % WARP_SIZE == 0) {
    gpuAtomicAdd(&dalpha_p[0], thread_dalpha_p);
    gpuAtomicAdd(&dalpha_n[0], thread_dalpha_n);
  }
}

torch::Tensor XIELUAutograd::forward(AutogradContext *ctx, Tensor x,
                                     Tensor alpha_p, Tensor alpha_n,
                                     double beta, double eps) {

  PUSH_RANGE("XIELU_FWD", 0)

  TORCH_CHECK(x.is_cuda(), "Input tensor x must be on the CUDA device.");
  TORCH_CHECK(alpha_p.is_cuda(),
              "Input tensor alpha_p must be on the CUDA device.");
  TORCH_CHECK(alpha_n.is_cuda(),
              "Input tensor alpha_n must be on the CUDA device.");
  TORCH_CHECK(alpha_p.dim() == 1 && alpha_p.numel() == 1,
              "alpha_p must be a 1-D tensor with one element.");
  TORCH_CHECK(alpha_n.dim() == 1 && alpha_n.numel() == 1,
              "alpha_n must be a 1-D tensor with one element.");
  TORCH_CHECK(x.dtype() == alpha_p.dtype(), "Data type of x (", x.dtype(),
              ") must match data type of alpha_p (", alpha_p.dtype(), ").");
  TORCH_CHECK(x.dtype() == alpha_n.dtype(), "Data type of x (", x.dtype(),
              ") must match data type of alpha_n (", alpha_n.dtype(), ").");

  const int batch_size = x.size(0);
  const int seq_len = x.size(1);
  const int hidden_dim = x.size(2);

  const int blockSize = NWARPS * WARP_SIZE;
  const int numBlocks =
      max(1, min(getMaxBlocks(),
                 ((batch_size * seq_len * hidden_dim) + blockSize - 1) /
                     blockSize));

  TensorOptions options = x.options();
  Tensor output = torch::empty_like(x);

  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const c10::cuda::CUDAStreamGuard guard(stream);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "forward", ([&] {
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

  TORCH_CHECK(x.is_cuda(), "Input tensor x must be on the CUDA device.");
  TORCH_CHECK(alpha_p.is_cuda(),
              "Input tensor alpha_p must be on the CUDA device.");
  TORCH_CHECK(alpha_n.is_cuda(),
              "Input tensor alpha_n must be on the CUDA device.");
  TORCH_CHECK(alpha_p.dim() == 1 && alpha_p.numel() == 1,
              "alpha_p must be a 1-D tensor with one element.");
  TORCH_CHECK(alpha_n.dim() == 1 && alpha_n.numel() == 1,
              "alpha_n must be a 1-D tensor with one element.");
  TORCH_CHECK(x.dtype() == alpha_p.dtype(), "Data type of x (", x.dtype(),
              ") must match data type of alpha_p (", alpha_p.dtype(), ").");
  TORCH_CHECK(x.dtype() == alpha_n.dtype(), "Data type of x (", x.dtype(),
              ") must match data type of alpha_n (", alpha_n.dtype(), ").");

  const int nbatch = x.size(0);
  const int seq_len = x.size(1);
  const int hidden_dim = x.size(2);

  const int blockSize = NWARPS * WARP_SIZE;
  const int numBlocks = max(
      1, min(getMaxBlocks(),
             ((nbatch * seq_len * hidden_dim) + blockSize - 1) / blockSize));

  TensorOptions options = x.options();

  Tensor dx = torch::empty_like(x);
  // each block stores a contribution to dalpha_p, dalpha_n
  // do reductions to dalpha_p, dalpha_n in higher precision to avoid numerical
  // errors
  Tensor dalpha_p;
  Tensor dalpha_n;

  if (x.scalar_type() == at::ScalarType::Half ||
      x.scalar_type() == at::ScalarType::BFloat16) {
    dalpha_p =
        torch::empty({1}, options.dtype(torch::kFloat32));
    dalpha_n =
        torch::empty({1}, options.dtype(torch::kFloat32));
  } else {
    dalpha_p = torch::empty({1}, options);
    dalpha_n = torch::empty({1}, options);
  }

  /*might not be needed - can check performance with/without.
  (contiguity isn't guaranteed for grad_outputs!)
  */

  if (!grad_outputs[0].is_contiguous())
    grad_outputs[0] = grad_outputs[0].contiguous();

  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const c10::cuda::CUDAStreamGuard guard(stream);
  if (x.scalar_type() == at::ScalarType::Half ||
      x.scalar_type() == at::ScalarType::BFloat16) {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
        "backward", ([&] {
          backward_kernel<scalar_t, float><<<numBlocks, blockSize, 0, stream>>>(
              get_accessor<scalar_t, 3>(x), get_accessor<scalar_t, 1>(alpha_p),
              get_accessor<scalar_t, 1>(alpha_n),
              get_accessor<scalar_t, 3>(grad_outputs[0]), (scalar_t)beta,
              (scalar_t)eps, get_accessor<scalar_t, 3>(dx),
              get_accessor<float, 1>(dalpha_p),
              get_accessor<float, 1>(dalpha_n));
        }));
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        x.scalar_type(), "backward", ([&] {
          backward_kernel<scalar_t, scalar_t>
              <<<numBlocks, blockSize, 0, stream>>>(
                  get_accessor<scalar_t, 3>(x),
                  get_accessor<scalar_t, 1>(alpha_p),
                  get_accessor<scalar_t, 1>(alpha_n),
                  get_accessor<scalar_t, 3>(grad_outputs[0]), (scalar_t)beta,
                  (scalar_t)eps, get_accessor<scalar_t, 3>(dx),
                  get_accessor<scalar_t, 1>(dalpha_p),
                  get_accessor<scalar_t, 1>(dalpha_n));
        }));
  }

  torch::Tensor dalpha_p_sum = dalpha_p.to(dx.dtype());
  torch::Tensor dalpha_n_sum = dalpha_n.to(dx.dtype());

  torch::Tensor undef;

  POP_RANGE

  return {dx, dalpha_p_sum, dalpha_n_sum, undef, undef};
}
