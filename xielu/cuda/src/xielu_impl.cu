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

/* specialized structure for vectorised loads with half, bfloat16 types */
template <typename T> struct vec4 { T x, y, z, w; };

using half4 = vec4<c10::Half>;
using bfloat4 = vec4<c10::BFloat16>;

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

// Generic overload: cast to float
template <typename T> __device__ __forceinline__ float to_float_if_needed(T x) {
  return static_cast<float>(x);
}

// Overload for float: return as-is
__device__ __forceinline__ float to_float_if_needed(float x) { return x; }

// Overload for double: return as-is
__device__ __forceinline__ double to_float_if_needed(double x) { return x; }

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_expm1(const scalar_t x) {
  float x_f = to_float_if_needed(x);
  return static_cast<scalar_t>(expf(x_f) - 1.0f);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute(scalar_t v,
                                            const scalar_t s_alpha_p,
                                            const scalar_t s_alpha_n,
                                            const scalar_t beta,
                                            const scalar_t eps) {

  return to_float_if_needed(v) > scalar_t(0.0)
             ? v * (s_alpha_p * v + beta)
             : (beta + s_alpha_n) * compute_expm1<scalar_t>(min(v, eps)) -
                   s_alpha_n * v;
}

template <typename scalar_t, typename vector_t>
__global__ void vectorised_xielu_forward_impl(
    const scalar_t *__restrict__ x, const int total_elements,
    const Accessor<scalar_t, 1> alpha_p, const Accessor<scalar_t, 1> alpha_n,
    const scalar_t beta, const scalar_t eps, scalar_t *__restrict__ output) {
  using sp = softplus<scalar_t>;
  const scalar_t s_alpha_p = sp::f(alpha_p[0]);
  const scalar_t s_alpha_n = sp::f(alpha_n[0]);

  for (int idx = blockIdx.x * blockDim.x + threadIdx.x * 4;
       idx < total_elements; idx += blockDim.x * gridDim.x * 4) {

    vector_t x_v = *reinterpret_cast<const vector_t *>(&x[idx]);
    vector_t out;

    out.x = compute(x_v.x, s_alpha_p, s_alpha_n, beta, eps);
    out.y = compute(x_v.y, s_alpha_p, s_alpha_n, beta, eps);
    out.z = compute(x_v.z, s_alpha_p, s_alpha_n, beta, eps);
    out.w = compute(x_v.w, s_alpha_p, s_alpha_n, beta, eps);

    *reinterpret_cast<vector_t *>(&output[idx]) = out;
  }
}

template <typename scalar_t>
__global__ void
xielu_forward_impl(const scalar_t *__restrict__ x, const int total_elements,
                   const Accessor<scalar_t, 1> alpha_p,
                   const Accessor<scalar_t, 1> alpha_n, const scalar_t beta,
                   const scalar_t eps, scalar_t *__restrict__ output) {
  using sp = softplus<scalar_t>;
  const scalar_t s_alpha_p = sp::f(alpha_p[0]);
  const scalar_t s_alpha_n = sp::f(alpha_n[0]);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements;
       i += blockDim.x * gridDim.x) {
    scalar_t x_v = x[i];
    scalar_t out = compute(x_v, s_alpha_p, s_alpha_n, beta, eps);
    output[i] = out;
  }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_dx(scalar_t v, scalar_t dout,
                                               scalar_t s_alpha_p,
                                               scalar_t s_alpha_n,
                                               scalar_t beta, scalar_t eps) {
  return to_float_if_needed(v) > scalar_t(0.0)
             ? dout * (2 * s_alpha_p * v + beta)
             : dout * (s_alpha_n * compute_expm1<scalar_t>(min(v, eps)) + beta);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_dp(scalar_t v, scalar_t dout,
                                               scalar_t ds_alpha_p) {
  return to_float_if_needed(v) > scalar_t(0.0) ? dout * ds_alpha_p * v * v
                                               : scalar_t(0.0);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_dn(scalar_t v, scalar_t dout,
                                               scalar_t ds_alpha_n,
                                               scalar_t eps) {
  return to_float_if_needed(v) <= scalar_t(0.0)
             ? dout * ds_alpha_n * (compute_expm1<scalar_t>(min(v, eps)) - v)
             : scalar_t(0.0);
}

template <typename scalar_t, typename reduction_type, typename vector_t>
__global__ void vectorised_xielu_backward_impl(
    const scalar_t *__restrict__ x, const int total_elements,
    const Accessor<scalar_t, 1> alpha_p, const Accessor<scalar_t, 1> alpha_n,
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

  reduction_type thread_dalpha_p = reduction_type(0.0);
  reduction_type thread_dalpha_n = reduction_type(0.0);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x * 4; i < total_elements;
       i += blockDim.x * gridDim.x * 4) {

    vector_t x_v = *reinterpret_cast<const vector_t *>(&x[i]);

    vector_t grad_output_v =
        *reinterpret_cast<const vector_t *>(&grad_outputs[i]);
    vector_t dx_v;
    vector_t dalpha_p_v, dalpha_n_v;

    dx_v.x =
        compute_dx(x_v.x, grad_output_v.x, s_alpha_p, s_alpha_n, beta, eps);
    dx_v.y =
        compute_dx(x_v.y, grad_output_v.y, s_alpha_p, s_alpha_n, beta, eps);
    dx_v.z =
        compute_dx(x_v.z, grad_output_v.z, s_alpha_p, s_alpha_n, beta, eps);
    dx_v.w =
        compute_dx(x_v.w, grad_output_v.w, s_alpha_p, s_alpha_n, beta, eps);

    dalpha_p_v.x = compute_dp(x_v.x, grad_output_v.x, ds_alpha_p);
    dalpha_p_v.y = compute_dp(x_v.y, grad_output_v.y, ds_alpha_p);
    dalpha_p_v.z = compute_dp(x_v.z, grad_output_v.z, ds_alpha_p);
    dalpha_p_v.w = compute_dp(x_v.w, grad_output_v.w, ds_alpha_p);

    dalpha_n_v.x = compute_dn(x_v.x, grad_output_v.x, ds_alpha_n, eps);
    dalpha_n_v.y = compute_dn(x_v.y, grad_output_v.y, ds_alpha_n, eps);
    dalpha_n_v.z = compute_dn(x_v.z, grad_output_v.z, ds_alpha_n, eps);
    dalpha_n_v.w = compute_dn(x_v.w, grad_output_v.w, ds_alpha_n, eps);

    *reinterpret_cast<vector_t *>(&dx[i]) = dx_v;

    thread_dalpha_p +=
        dalpha_p_v.x + dalpha_p_v.y + dalpha_p_v.z + dalpha_p_v.w;
    thread_dalpha_n +=
        dalpha_n_v.x + dalpha_n_v.y + dalpha_n_v.z + dalpha_n_v.w;
  }

  __syncthreads();

  // reduce thread-local contributions into thread % 32 = 0
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    thread_dalpha_p += __shfl_down_sync(0xffffffff, thread_dalpha_p, offset);
    thread_dalpha_n += __shfl_down_sync(0xffffffff, thread_dalpha_n, offset);
  }

  // write each warp's contributions to grad to gmem
  if (threadIdx.x % WARP_SIZE == 0) {
    gpuAtomicAdd(&dalpha_p[0], thread_dalpha_p);
    gpuAtomicAdd(&dalpha_n[0], thread_dalpha_n);
  }
}

template <typename scalar_t, typename reduction_type>
__global__ void xielu_backward_impl(const scalar_t *__restrict__ x,
                                    const int total_elements,
                                    const Accessor<scalar_t, 1> alpha_p,
                                    const Accessor<scalar_t, 1> alpha_n,
                                    const scalar_t *__restrict__ grad_outputs,
                                    const scalar_t beta, const scalar_t eps,
                                    scalar_t *__restrict__ dx,
                                    Accessor<reduction_type, 1> dalpha_p,
                                    Accessor<reduction_type, 1> dalpha_n) {

  using sp = softplus<scalar_t>;
  const scalar_t _alpha_p = alpha_p[0];
  const scalar_t _alpha_n = alpha_n[0];

  const scalar_t s_alpha_p = sp::f(_alpha_p);
  const scalar_t s_alpha_n = beta + sp::f(_alpha_n);
  const scalar_t ds_alpha_p = sp::df(_alpha_p);
  const scalar_t ds_alpha_n = sp::df(_alpha_n);

  reduction_type thread_dalpha_p = reduction_type(0.0);
  reduction_type thread_dalpha_n = reduction_type(0.0);

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements;
       i += blockDim.x * gridDim.x) {

    scalar_t x_v = x[i];
    scalar_t grad_output_v = grad_outputs[i];

    scalar_t dx_v =
        compute_dx(x_v, grad_output_v, s_alpha_p, s_alpha_n, beta, eps);
    scalar_t dalpha_p_v = compute_dp(x_v, grad_output_v, ds_alpha_p);
    scalar_t dalpha_n_v = compute_dn(x_v, grad_output_v, ds_alpha_n, eps);

    dx[i] = dx_v;

    thread_dalpha_p += dalpha_p_v;
    thread_dalpha_n += dalpha_n_v;
  }

  __syncthreads();

  // reduce thread-local contributions into thread % 32 = 0
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    thread_dalpha_p += __shfl_down_sync(0xffffffff, thread_dalpha_p, offset);
    thread_dalpha_n += __shfl_down_sync(0xffffffff, thread_dalpha_n, offset);
  }

  // write each warp's contributions to grad to gmem
  if (threadIdx.x % WARP_SIZE == 0) {
    gpuAtomicAdd(&dalpha_p[0], thread_dalpha_p);
    gpuAtomicAdd(&dalpha_n[0], thread_dalpha_n);
  }
}

torch::Tensor XIELUAutograd::forward(AutogradContext *ctx, Tensor x,
                                     Tensor alpha_p, Tensor alpha_n,
                                     double beta, double eps,
                                     bool with_vector_loads) {

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
  const int nelements = batch_size * seq_len * hidden_dim;

  TORCH_CHECK(hidden_dim % 4 == 0, "hidden_dim must be a multiple of 4");

  const int blockSize = NWARPS * WARP_SIZE;
  const int numBlocks =
      max(1, min(getMaxBlocks(), (nelements + blockSize - 1) / blockSize));

  TensorOptions options = x.options();
  Tensor output = torch::empty_like(x);

  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const c10::cuda::CUDAStreamGuard guard(stream);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "forward", ([&] {
        using vector_t = typename std::conditional<
            std::is_same<scalar_t, float>::value, float4,
            typename std::conditional<
                std::is_same<scalar_t, double>::value, double4,
                typename std::conditional<
                    std::is_same<scalar_t, c10::Half>::value, half4,
                    typename std::conditional<
                        std::is_same<scalar_t, c10::BFloat16>::value, bfloat4,
                        void>::type>::type>::type>::type;

        static_assert(!std::is_same<vector_t, void>::value, "Unsupported type");

        if (with_vector_loads) {
          vectorised_xielu_forward_impl<scalar_t, vector_t>
              <<<numBlocks, blockSize, 0, stream>>>(
                  x.data_ptr<scalar_t>(), batch_size * seq_len * hidden_dim,
                  get_accessor<scalar_t, 1>(alpha_p),
                  get_accessor<scalar_t, 1>(alpha_n), (scalar_t)beta,
                  (scalar_t)eps, output.data_ptr<scalar_t>());
        } else {

          xielu_forward_impl<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
              x.data_ptr<scalar_t>(), batch_size * seq_len * hidden_dim,
              get_accessor<scalar_t, 1>(alpha_p),
              get_accessor<scalar_t, 1>(alpha_n), (scalar_t)beta, (scalar_t)eps,
              output.data_ptr<scalar_t>());
        }
      }));

  ctx->save_for_backward({x, alpha_p, alpha_n});
  ctx->saved_data["eps"] = eps;
  ctx->saved_data["beta"] = beta;
  ctx->saved_data["with_vector_loads"] = with_vector_loads;

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
  const bool with_vector_loads = ctx->saved_data["with_vector_loads"].toBool();

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
  const int nelements = nbatch * seq_len * hidden_dim;

  TORCH_CHECK(hidden_dim % 4 == 0, "hidden_dim must be a multiple of 4");

  const int blockSize = NWARPS * WARP_SIZE;
  const int numBlocks =
      max(1, min(getMaxBlocks(), (nelements + blockSize - 1) / blockSize));
  TensorOptions options = x.options();

  Tensor dx = torch::empty_like(x);
  // each block stores a contribution to dalpha_p, dalpha_n
  // do reductions to dalpha_p, dalpha_n in higher precision to avoid
  // numerical errors
  Tensor dalpha_p;
  Tensor dalpha_n;

  if (x.scalar_type() == at::ScalarType::Half ||
      x.scalar_type() == at::ScalarType::BFloat16) {
    dalpha_p = torch::zeros({1}, options.dtype(torch::kFloat32));
    dalpha_n = torch::zeros({1}, options.dtype(torch::kFloat32));
  } else {
    dalpha_p = torch::zeros({1}, options);
    dalpha_n = torch::zeros({1}, options);
  }

  if (!grad_outputs[0].is_contiguous())
    grad_outputs[0] = grad_outputs[0].contiguous();

  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const c10::cuda::CUDAStreamGuard guard(stream);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "backward", ([&] {
        using vector_t = typename std::conditional<
            std::is_same<scalar_t, float>::value, float4,
            typename std::conditional<
                std::is_same<scalar_t, double>::value, double4,
                typename std::conditional<
                    std::is_same<scalar_t, c10::Half>::value, half4,
                    typename std::conditional<
                        std::is_same<scalar_t, c10::BFloat16>::value, bfloat4,
                        void>::type>::type>::type>::type;

        static_assert(!std::is_same<vector_t, void>::value, "Unsupported type");

        using reduction_t = typename std::conditional<
            std::is_same<scalar_t, c10::Half>::value ||
                std::is_same<scalar_t, c10::BFloat16>::value,
            float, scalar_t>::type;

        if (with_vector_loads) {
          vectorised_xielu_backward_impl<scalar_t, reduction_t, vector_t>
              <<<numBlocks, blockSize, 0, stream>>>(
                  x.data_ptr<scalar_t>(), nbatch * seq_len * hidden_dim,
                  get_accessor<scalar_t, 1>(alpha_p),
                  get_accessor<scalar_t, 1>(alpha_n),
                  grad_outputs[0].data_ptr<scalar_t>(), (scalar_t)beta,
                  (scalar_t)eps, dx.data_ptr<scalar_t>(),
                  get_accessor<reduction_t, 1>(dalpha_p),
                  get_accessor<reduction_t, 1>(dalpha_n));
        } else {
          xielu_backward_impl<scalar_t, reduction_t>
              <<<numBlocks, blockSize, 0, stream>>>(
                  x.data_ptr<scalar_t>(), nbatch * seq_len * hidden_dim,
                  get_accessor<scalar_t, 1>(alpha_p),
                  get_accessor<scalar_t, 1>(alpha_n),
                  grad_outputs[0].data_ptr<scalar_t>(), (scalar_t)beta,
                  (scalar_t)eps, dx.data_ptr<scalar_t>(),
                  get_accessor<reduction_t, 1>(dalpha_p),
                  get_accessor<reduction_t, 1>(dalpha_n));
        }
      }));

  torch::Tensor undef;

  POP_RANGE

  return {dx,   dalpha_p.to(x.dtype()), dalpha_n.to(x.dtype()), undef, undef,
          undef};
}
