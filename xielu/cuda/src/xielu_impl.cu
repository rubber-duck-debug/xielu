#include "utils.cuh"
#include "xielu_impl.hpp"

#include <iostream>

#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <torch/script.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;
using namespace c10;

using torch::Tensor;
using torch::TensorOptions;

#define NWARPS 8
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

template <typename scalar_t, typename vector_t>
__global__ void
forward_kernel(const scalar_t *__restrict__ x, const int total_elements,
               const Accessor<scalar_t, 1> alpha_p,
               const Accessor<scalar_t, 1> alpha_n, const scalar_t beta,
               const scalar_t eps, scalar_t *__restrict__ output) {
  using sp = softplus<scalar_t>;
  const scalar_t s_alpha_p = sp::f(alpha_p[0]);
  const scalar_t s_alpha_n = sp::f(alpha_n[0]);

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x * 4;
       idx < total_elements; idx += blockDim.x * gridDim.x * 4) {

    vector_t x_v = *reinterpret_cast<const vector_t *>(&x[idx]);
    vector_t out;
    out.x =
        x_v.x > scalar_t(0.0)
            ? s_alpha_p * x_v.x * x_v.x + beta * x_v.x
            : (beta + s_alpha_n) * expm1(min(x_v.x, eps)) - s_alpha_n * x_v.x;
    out.y =
        x_v.y > scalar_t(0.0)
            ? s_alpha_p * x_v.y * x_v.y + beta * x_v.y
            : (beta + s_alpha_n) * expm1(min(x_v.y, eps)) - s_alpha_n * x_v.y;
    out.z =
        x_v.z > scalar_t(0.0)
            ? s_alpha_p * x_v.z * x_v.z + beta * x_v.z
            : (beta + s_alpha_n) * expm1(min(x_v.z, eps)) - s_alpha_n * x_v.z;
    out.w =
        x_v.w > scalar_t(0.0)
            ? s_alpha_p * x_v.w * x_v.w + beta * x_v.w
            : (beta + s_alpha_n) * expm1(min(x_v.w, eps)) - s_alpha_n * x_v.w;

    *reinterpret_cast<vector_t *>(&output[idx]) = out;
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

  // AT_DISPATCH_FLOATING_TYPES_AND2(
  //     at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
  //     "forward", ([&] {
  //       forward_kernel<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
  //           x.data_ptr<scalar_t>(), batch_size * seq_len * hidden_dim,
  //           get_accessor<scalar_t, 1>(alpha_p),
  //           get_accessor<scalar_t, 1>(alpha_n), (scalar_t)beta,
  //           (scalar_t)eps, output.data_ptr<scalar_t>());
  //     }));

  AT_DISPATCH_FLOATING_TYPES(
      x.scalar_type(), "forward", ([&] {
        using vector_t = typename std::conditional<
            std::is_same<scalar_t, float>::value,
            float4, // Use float4 for float32
            typename std::conditional<std::is_same<scalar_t, double>::value,
                                      double4, // Use double4 for float64
                                      void     // Unsupported types will cause a
                                               // compile-time error
                                      >::type>::type;

        forward_kernel<scalar_t, vector_t><<<numBlocks, blockSize, 0, stream>>>(
            x.data_ptr<scalar_t>(), batch_size * seq_len * hidden_dim,
            get_accessor<scalar_t, 1>(alpha_p),
            get_accessor<scalar_t, 1>(alpha_n), (scalar_t)beta, (scalar_t)eps,
            output.data_ptr<scalar_t>());
      }));

  ctx->save_for_backward({x, alpha_p, alpha_n});
  ctx->saved_data["eps"] = eps;
  ctx->saved_data["beta"] = beta;

  POP_RANGE

  return output;
}

template <typename scalar_t, typename reduction_type, typename vector_t>
__global__ void
backward_kernel(const scalar_t *__restrict__ x, const int total_elements,
                const Accessor<scalar_t, 1> alpha_p,
                const Accessor<scalar_t, 1> alpha_n,
                const scalar_t *__restrict__ grad_outputs, const scalar_t beta,
                const scalar_t eps, scalar_t *__restrict__ dx,
                Accessor<reduction_type, 1> dalpha_p,
                Accessor<reduction_type, 1> dalpha_n) {

  using sp = softplus<scalar_t>;
  const scalar_t _alpha_p = alpha_p[0];
  const scalar_t _alpha_n = alpha_n[0];

  const scalar_t s_alpha_p = sp::f(_alpha_p);
  const scalar_t s_alpha_n = beta + sp::f(_alpha_n);
  const scalar_t ds_alpha_p = sp::df(_alpha_p);
  const scalar_t ds_alpha_n = sp::df(_alpha_n);

  reduction_type thread_dalpha_p = 0.0;
  reduction_type thread_dalpha_n = 0.0;

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x * 4;
       idx < total_elements; idx += blockDim.x * gridDim.x * 4) {

    vector_t x_v = *reinterpret_cast<const vector_t *>(&x[idx]);

    vector_t grad_output_v =
        *reinterpret_cast<const vector_t *>(&grad_outputs[idx]);
    vector_t dx_v;
    vector_t dalpha_p_v, dalpha_n_v;

    dx_v.x = x_v.x > scalar_t(0.0)
                 ? grad_output_v.x * (2 * s_alpha_p * x_v.x + beta)
                 : grad_output_v.x *
                       (s_alpha_n * expm1(min(x_v.x, eps)) + beta);
    dx_v.y = x_v.y > scalar_t(0.0)
                 ? grad_output_v.y * (2 * s_alpha_p * x_v.y + beta)
                 : grad_output_v.y *
                       (s_alpha_n * expm1(min(x_v.y, eps)) + beta);

    dx_v.z = x_v.z > scalar_t(0.0)
                 ? grad_output_v.z * (2 * s_alpha_p * x_v.z + beta)
                 : grad_output_v.z *
                       (s_alpha_n * expm1(min(x_v.z, eps)) + beta);

    dx_v.w = x_v.w > scalar_t(0.0)
                 ? grad_output_v.w * (2 * s_alpha_p * x_v.w + beta)
                 : grad_output_v.w *
                       (s_alpha_n * expm1(min(x_v.w, eps)) + beta);

    dalpha_p_v.x = x_v.x > scalar_t(0.0)
                       ? grad_output_v.x * ds_alpha_p * x_v.x * x_v.x
                       : scalar_t(0.0);
    dalpha_p_v.y = x_v.y > scalar_t(0.0)
                       ? grad_output_v.y * ds_alpha_p * x_v.y * x_v.y
                       : scalar_t(0.0);
    dalpha_p_v.z = x_v.z > scalar_t(0.0)
                       ? grad_output_v.z * ds_alpha_p * x_v.z * x_v.z
                       : scalar_t(0.0);
    dalpha_p_v.w = x_v.w > scalar_t(0.0)
                       ? grad_output_v.w * ds_alpha_p * x_v.w * x_v.w
                       : scalar_t(0.0);

    dalpha_n_v.x = x_v.x <= scalar_t(0.0) ? grad_output_v.x * ds_alpha_n *
                                                (expm1(min(x_v.x, eps)) - x_v.x)
                                          : scalar_t(0.0);
    dalpha_n_v.y = x_v.y <= scalar_t(0.0) ? grad_output_v.y * ds_alpha_n *
                                                (expm1(min(x_v.y, eps)) - x_v.y)
                                          : scalar_t(0.0);
    dalpha_n_v.z = x_v.z <= scalar_t(0.0) ? grad_output_v.z * ds_alpha_n *
                                                (expm1(min(x_v.z, eps)) - x_v.z)
                                          : scalar_t(0.0);
    dalpha_n_v.w = x_v.w <= scalar_t(0.0) ? grad_output_v.w * ds_alpha_n *
                                                (expm1(min(x_v.w, eps)) - x_v.w)
                                          : scalar_t(0.0);

    *reinterpret_cast<vector_t *>(&dx[idx]) = dx_v;

    thread_dalpha_p +=
        dalpha_p_v.x + dalpha_p_v.y + dalpha_p_v.z + dalpha_p_v.w;
    thread_dalpha_n +=
        dalpha_n_v.x + dalpha_n_v.y + dalpha_n_v.z + dalpha_n_v.w;
    /*
      if (static_cast<float>(x_e) > 0.0f) {
        dx[idx] = grad_output * (2 * s_alpha_p * x_e + beta);
        thread_dalpha_p += grad_output * sp::df(_alpha_p) * x_e * x_e;
      } else {
        dx[idx] =
            grad_output * ((beta + s_alpha_n) * exp(min(x_e, eps)) - s_alpha_n);
        thread_dalpha_n +=
            grad_output * sp::df(_alpha_n) * (expm1(min(x_e, eps)) - x_e);
      }
      */
  }

  __syncthreads();

  // reduce thread-local contributions into thread % 32 = 0
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    thread_dalpha_p += __shfl_down_sync(0xFFFFFFFF, thread_dalpha_p, offset);
    thread_dalpha_n += __shfl_down_sync(0xFFFFFFFF, thread_dalpha_n, offset);
  }

  // write each warp's contributions to grad to gmem
  if (threadIdx.x % WARP_SIZE == 0) {
    dalpha_p[blockIdx.x * NWARPS + (threadIdx.x / WARP_SIZE)] = thread_dalpha_p;
    dalpha_n[blockIdx.x * NWARPS + (threadIdx.x / WARP_SIZE)] = thread_dalpha_n;
  }
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
        torch::empty({numBlocks * NWARPS}, options.dtype(torch::kFloat32));
    dalpha_n =
        torch::empty({numBlocks * NWARPS}, options.dtype(torch::kFloat32));
  } else {
    dalpha_p = torch::empty({numBlocks * NWARPS}, options);
    dalpha_n = torch::empty({numBlocks * NWARPS}, options);
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
    /*AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
        "backward", ([&] {
          backward_kernel<scalar_t, float><<<numBlocks, blockSize, 0, stream>>>(
              x.data_ptr<scalar_t>(), nbatch * seq_len * hidden_dim,
       get_accessor<scalar_t, 1>(alpha_p), get_accessor<scalar_t, 1>(alpha_n),
              grad_outputs[0].data_ptr<scalar_t>(), (scalar_t)beta,
              (scalar_t)eps, dx.data_ptr<scalar_t>(),
              get_accessor<float, 1>(dalpha_p),
              get_accessor<float, 1>(dalpha_n));
        })); */
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        x.scalar_type(), "backward", ([&] {
          using vector_t = typename std::conditional<
              std::is_same<scalar_t, float>::value,
              float4, // Use float4 for float32
              typename std::conditional<std::is_same<scalar_t, double>::value,
                                        double4, // Use double4 for float64
                                        void // Unsupported types will cause a
                                             // compile-time error
                                        >::type>::type;

          backward_kernel<scalar_t, scalar_t, vector_t>
              <<<numBlocks, blockSize, 0, stream>>>(
                  x.data_ptr<scalar_t>(), nbatch * seq_len * hidden_dim,
                  get_accessor<scalar_t, 1>(alpha_p),
                  get_accessor<scalar_t, 1>(alpha_n),
                  grad_outputs[0].data_ptr<scalar_t>(), (scalar_t)beta,
                  (scalar_t)eps, dx.data_ptr<scalar_t>(),
                  get_accessor<scalar_t, 1>(dalpha_p),
                  get_accessor<scalar_t, 1>(dalpha_n));
        }));
  }

  torch::Tensor dalpha_p_sum = torch::sum(dalpha_p, 0, true).to(dx.dtype());
  torch::Tensor dalpha_n_sum = torch::sum(dalpha_n, 0, true).to(dx.dtype());

  torch::Tensor undef;

  POP_RANGE

  return {dx, dalpha_p_sum, dalpha_n_sum, undef, undef};
}
