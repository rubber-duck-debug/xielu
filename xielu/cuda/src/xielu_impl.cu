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
  return numMultiprocessors * 8;
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

template <typename scalar_t> struct VectorIO;

template <> struct VectorIO<float> {
  using scalar_t = float;
  using native_t = float;
  using vec_t = float4;
  using reduction_t = float;

  const static int packed_size = 4;

  __device__ static void unpack(const vec_t &v, scalar_t &x0, scalar_t &x1,
                                scalar_t &x2, scalar_t &x3) {
    x0 = v.x;
    x1 = v.y;
    x2 = v.z;
    x3 = v.w;
  }

  __device__ static vec_t pack(scalar_t x0, scalar_t x1, scalar_t x2,
                               scalar_t x3) {
    return {x0, x1, x2, x3};
  }
};

template <> struct VectorIO<double> {
  using scalar_t = double;
  using native_t = double;
  using vec_t = double4;
  using reduction_t = double;

  const static int packed_size = 4;

  __device__ static void unpack(const vec_t &v, scalar_t &x0, scalar_t &x1,
                                scalar_t &x2, scalar_t &x3) {
    x0 = v.x;
    x1 = v.y;
    x2 = v.z;
    x3 = v.w;
  }

  __device__ static vec_t pack(scalar_t x0, scalar_t x1, scalar_t x2,
                               scalar_t x3) {
    return {x0, x1, x2, x3};
  }
};

template <> struct VectorIO<c10::Half> {
  using scalar_t = c10::Half;
  using native_t = __half;
  using vec_t = __half2;
  using reduction_t = float;

  const static int packed_size = 2;

  __device__ static void unpack2(const vec_t &v, scalar_t &x0, scalar_t &x1) {
    const native_t *ptr = reinterpret_cast<const native_t *>(&v);
    x0 = static_cast<scalar_t>(ptr[0]);
    x1 = static_cast<scalar_t>(ptr[1]);
  }

  __device__ static vec_t pack2(scalar_t x0, scalar_t x1) {
    vec_t v;
    native_t *ptr = reinterpret_cast<native_t *>(&v);
    ptr[0] = static_cast<native_t>(x0);
    ptr[1] = static_cast<native_t>(x1);
    return v;
  }
};

template <> struct VectorIO<c10::BFloat16> {
  using scalar_t = c10::BFloat16;
  using native_t = __nv_bfloat16;
  using vec_t = __nv_bfloat162;
  using reduction_t = float;

  const static int packed_size = 2;

  __device__ static void unpack2(const vec_t &v, scalar_t &x0, scalar_t &x1) {
    const native_t *ptr = reinterpret_cast<const native_t *>(&v);
    x0 = static_cast<scalar_t>(ptr[0]);
    x1 = static_cast<scalar_t>(ptr[1]);
  }

  __device__ static vec_t pack2(scalar_t x0, scalar_t x1) {
    vec_t v;
    native_t *ptr = reinterpret_cast<native_t *>(&v);
    ptr[0] = static_cast<native_t>(x0);
    ptr[1] = static_cast<native_t>(x1);
    return v;
  }
};

// Generic overload: cast to float
template <typename T> __device__ __forceinline__ float to_float_if_needed(T x) {
  return static_cast<float>(x);
}

// Overload for __half
__device__ __forceinline__ float to_float_if_needed(__half x) {
  return __half2float(x);
}

// Overload for __nv_bfloat16
__device__ __forceinline__ float to_float_if_needed(__nv_bfloat16 x) {
  return __bfloat162float(x);
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
__device__ scalar_t compute(scalar_t v, const scalar_t s_alpha_p,
                            const scalar_t s_alpha_n, const scalar_t beta,
                            const scalar_t eps) {

  return to_float_if_needed(v) > scalar_t(0.0)
             ? v * (s_alpha_p * v + beta)
             : (beta + s_alpha_n) * compute_expm1<scalar_t>(min(v, eps)) -
                   s_alpha_n * v;
}

template <typename T> __device__ __forceinline__ T select2(T cond, T a, T b);

template <>
__device__ __forceinline__ __nv_bfloat162 select2<__nv_bfloat162>(
    __nv_bfloat162 cond, __nv_bfloat162 a, __nv_bfloat162 b) {
  __nv_bfloat16 cond0 = __low2bfloat16(cond);
  __nv_bfloat16 cond1 = __high2bfloat16(cond);
  __nv_bfloat16 a0 = __low2bfloat16(a);
  __nv_bfloat16 a1 = __high2bfloat16(a);
  __nv_bfloat16 b0 = __low2bfloat16(b);
  __nv_bfloat16 b1 = __high2bfloat16(b);

  __nv_bfloat16 out0 = cond0 != __float2bfloat16_rn(0.0f) ? a0 : b0;
  __nv_bfloat16 out1 = cond1 != __float2bfloat16_rn(0.0f) ? a1 : b1;

  return __halves2bfloat162(out0, out1);
}

template <>
__device__ __forceinline__ __half2 select2<__half2>(__half2 cond, __half2 a,
                                                    __half2 b) {
  __half cond0 = __low2half(cond);
  __half cond1 = __high2half(cond);
  __half a0 = __low2half(a);
  __half a1 = __high2half(a);
  __half b0 = __low2half(b);
  __half b1 = __high2half(b);

  __half zero = __float2half(0.0f);
  __half out0 = cond0 != zero ? a0 : b0;
  __half out1 = cond1 != zero ? a1 : b1;

  return __halves2half2(out0, out1);
}

// Type traits and math wrappers for __half2
__device__ __forceinline__ __half2 zero2(__half2) {
  return __float2half2_rn(0.0f);
}
__device__ __forceinline__ __half2 one2(__half2) {
  return __float2half2_rn(1.0f);
}
__device__ __forceinline__ __half2 two2(__half2) {
  return __float2half2_rn(2.0f);
}

// Type traits and math wrappers for __nv_bfloat162
__device__ __forceinline__ __nv_bfloat162 zero2(__nv_bfloat162) {
  return __float2bfloat162_rn(0.0f);
}
__device__ __forceinline__ __nv_bfloat162 one2(__nv_bfloat162) {
  return __float2bfloat162_rn(1.0f);
}
__device__ __forceinline__ __nv_bfloat162 two2(__nv_bfloat162) {
  return __float2bfloat162_rn(2.0f);
}

template <typename T>
__device__ __forceinline__ T compute_f16(T v, const T s_alpha_p,
                                         const T s_alpha_n, const T beta,
                                         const T eps, const T beta_s_alpha_n) {
  T zero = zero2(v);
  T one = one2(v);
  T exp_input = __hmin2(v, eps);

  T expm1_val = h2exp(exp_input);
  expm1_val = __hsub2(expm1_val, one);

  T pos = __hmul2(__hadd2(__hmul2(s_alpha_p, v), beta), v);
  T neg = __hsub2(__hmul2(beta_s_alpha_n, expm1_val), __hmul2(s_alpha_n, v));

  T mask = __hgt2(v, zero);
  return select2(mask, pos, neg);
}

template <typename scalar_t, const int nelements_per_thread>
__global__ void vectorised_xielu_forward_impl(
    const scalar_t *__restrict__ x, int total_elements,
    const Accessor<scalar_t, 1> alpha_p, const Accessor<scalar_t, 1> alpha_n,
    scalar_t beta, scalar_t eps, scalar_t *__restrict__ output) {

  using Traits = VectorIO<scalar_t>;
  using vec_t = typename Traits::vec_t;
  using native_t = typename Traits::native_t;
  const int packed_size =
      Traits::packed_size; // how many elements are packed into the vec_t

  const scalar_t s_alpha_p = softplus<scalar_t>::f(alpha_p[0]);
  const scalar_t s_alpha_n = softplus<scalar_t>::f(alpha_n[0]);
  const scalar_t beta_s_alpha_n = beta + s_alpha_n;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  vec_t s_alpa_p2;
  vec_t s_alpa_n2;
  vec_t beta2;
  vec_t eps2;
  vec_t beta_s_alpha_n2;

  if constexpr (std::is_same<native_t, __nv_bfloat16>::value) {
    s_alpa_p2 = __float2bfloat162_rn(s_alpha_p);
    s_alpa_n2 = __float2bfloat162_rn(s_alpha_n);
    beta2 = __float2bfloat162_rn(beta);
    eps2 = __float2bfloat162_rn(eps);
    beta_s_alpha_n2 = __float2bfloat162_rn(beta_s_alpha_n);
  } else if constexpr (std::is_same<native_t, __half>::value) {
    s_alpa_p2 = __float2half2_rn(s_alpha_p);
    s_alpa_n2 = __float2half2_rn(s_alpha_n);
    beta2 = __float2half2_rn(beta);
    eps2 = __float2half2_rn(eps);
    beta_s_alpha_n2 = __float2half2_rn(beta_s_alpha_n);
  }

  for (int i = idx; i < total_elements / (packed_size * nelements_per_thread);
       i += stride) {
    int base_idx = i * (packed_size * nelements_per_thread);

    if constexpr (std::is_same<native_t, __nv_bfloat16>::value ||
                  std::is_same<native_t, __half>::value) {
      // bfloat162 + half2 specialized handling
#pragma unroll
      for (int j = 0; j < nelements_per_thread; ++j) {
        vec_t x_v =
            reinterpret_cast<const vec_t &>(x[base_idx + j * packed_size]);
        vec_t o_v = compute_f16(x_v, s_alpa_p2, s_alpa_n2, beta2, eps2,
                                beta_s_alpha_n2);
        reinterpret_cast<vec_t &>(output[base_idx + j * packed_size]) = o_v;
      }
    } else {
// Generic scalar path (float, double)
#pragma unroll
      for (int j = 0; j < nelements_per_thread; ++j) {

        vec_t x_vec =
            *reinterpret_cast<const vec_t *>(&x[base_idx + j * packed_size]);
        scalar_t in0, in1, in2, in3;

        Traits::unpack(x_vec, in0, in1, in2, in3);

        scalar_t out0 = compute(in0, s_alpha_p, s_alpha_n, beta, eps);
        scalar_t out1 = compute(in1, s_alpha_p, s_alpha_n, beta, eps);
        scalar_t out2 = compute(in2, s_alpha_p, s_alpha_n, beta, eps);
        scalar_t out3 = compute(in3, s_alpha_p, s_alpha_n, beta, eps);

        vec_t out_vec = Traits::pack(out0, out1, out2, out3);
        *reinterpret_cast<vec_t *>(&output[base_idx + j * packed_size]) =
            out_vec;
      }
    }
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

  TORCH_CHECK(hidden_dim % 8 == 0, "hidden_dim must be a multiple of 4");

  const int blockSize = NWARPS * WARP_SIZE;

  TensorOptions options = x.options();
  Tensor output = torch::empty_like(x);

  const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
  const c10::cuda::CUDAStreamGuard guard(stream);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(),
      "forward", ([&] {
        // half2/bfloat162, 2 elements per var, 4 loads per thread -> 8 total
        // elements
        // float4/double4, 4 elements per var, 2 loads per thread -> 8 total
        using Traits = VectorIO<scalar_t>;

        const int loads_per_thread =
            with_vector_loads ? (Traits::packed_size == 2 ? 4 : 1) : 1;

        const int elements_per_thread =
            with_vector_loads ? loads_per_thread * Traits::packed_size : 1;

        const int adjusted_elements = nelements / elements_per_thread;
        const int numBlocks =
            max(1, min(getMaxBlocks(),
                       (adjusted_elements + blockSize - 1) / blockSize));

        if (with_vector_loads) {
          vectorised_xielu_forward_impl<scalar_t,
                                        (Traits::packed_size == 2 ? 4 : 1)>
              <<<numBlocks, blockSize, 0, stream>>>(
                  x.data_ptr<scalar_t>(), nelements,
                  get_accessor<scalar_t, 1>(alpha_p),
                  get_accessor<scalar_t, 1>(alpha_n), (scalar_t)beta,
                  (scalar_t)eps, output.data_ptr<scalar_t>());
        } else {

          xielu_forward_impl<scalar_t><<<numBlocks, blockSize, 0, stream>>>(
              x.data_ptr<scalar_t>(), nelements,
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

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_dx(scalar_t v, scalar_t dout,
                                               scalar_t s_alpha_p,
                                               scalar_t s_alpha_n,
                                               scalar_t beta, scalar_t eps) {
  return to_float_if_needed(v) > scalar_t(0.0)
             ? dout * (2 * s_alpha_p * v + beta)
             : dout * (s_alpha_n * compute_expm1<scalar_t>(min(v, eps)) + beta);
}

template <typename T>
__device__ __forceinline__ T compute_dx_f16(T v, T dout, T s_alpha_p,
                                            T s_alpha_n, T beta, T eps) {
  T zero = zero2(v);
  T one = one2(v);
  T two = two2(v);
  T exp_input = __hmin2(v, eps);
  T expm1_val = __hsub2(h2exp(exp_input), one);

  T pos = __hmul2(dout, __hadd2(__hmul2(__hmul2(s_alpha_p, v), two), beta));
  T neg = __hmul2(dout, __hadd2(__hmul2(s_alpha_n, expm1_val), beta));

  T mask = __hgt2(v, zero);
  return select2(mask, pos, neg);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_dp(scalar_t v, scalar_t dout,
                                               scalar_t ds_alpha_p) {
  return to_float_if_needed(v) > scalar_t(0.0) ? dout * ds_alpha_p * v * v
                                               : scalar_t(0.0);
}

template <typename T>
__device__ __forceinline__ T compute_dp_f16(T v, T dout, T ds_alpha_p) {
  T zero = zero2(v);
  T mask = __hgt2(v, zero);

  T term = __hmul2(dout, __hmul2(ds_alpha_p, __hmul2(v, v)));
  return select2(mask, term, zero);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_dn(scalar_t v, scalar_t dout,
                                               scalar_t ds_alpha_n,
                                               scalar_t eps) {
  return to_float_if_needed(v) <= scalar_t(0.0)
             ? dout * ds_alpha_n * (compute_expm1<scalar_t>(min(v, eps)) - v)
             : scalar_t(0.0);
}

template <typename T>
__device__ __forceinline__ T compute_dn_f16(T v, T dout, T ds_alpha_n, T eps) {
  T zero = zero2(v);
  T one = one2(v);
  T exp_input = __hmin2(v, eps);
  T expm1_val = __hsub2(h2exp(exp_input), one);

  T term = __hmul2(dout, __hmul2(ds_alpha_n, __hsub2(expm1_val, v)));
  T mask = __hle2(v, zero);

  return select2(mask, term, zero);
}

__device__ __forceinline__ float float16_sum(__nv_bfloat162 val) {
  float2 v = __bfloat1622float2(val);
  return v.x + v.y;
}

__device__ __forceinline__ float float16_sum(__half2 val) {
  float2 v = __half22float2(val);
  return v.x + v.y;
}

template <typename scalar_t, typename reduction_type,
          const int nelements_per_thread>
__global__ void vectorised_xielu_backward_impl(
    const scalar_t *__restrict__ x, const int total_elements,
    const Accessor<scalar_t, 1> alpha_p, const Accessor<scalar_t, 1> alpha_n,
    const scalar_t *__restrict__ grad_outputs, const scalar_t beta,
    const scalar_t eps, scalar_t *__restrict__ dx,
    Accessor<reduction_type, 1> dalpha_p,
    Accessor<reduction_type, 1> dalpha_n) {

  using Traits = VectorIO<scalar_t>;
  using vec_t = typename Traits::vec_t;
  using native_t = typename Traits::native_t;
  const int packed_size =
      Traits::packed_size; // how many elements are packed into the vec_t

  using sp = softplus<scalar_t>;
  const scalar_t _alpha_p = alpha_p[0];
  const scalar_t _alpha_n = alpha_n[0];

  const scalar_t s_alpha_p = sp::f(_alpha_p);
  const scalar_t s_alpha_n = beta + sp::f(_alpha_n);
  const scalar_t ds_alpha_p = sp::df(_alpha_p);
  const scalar_t ds_alpha_n = sp::df(_alpha_n);

  vec_t s_alpa_p2;
  vec_t ds_alpha_p2;
  vec_t s_alpa_n2;
  vec_t ds_alpha_n2;
  vec_t beta2;
  vec_t eps2;

  if constexpr (std::is_same<native_t, __nv_bfloat16>::value) {
    s_alpa_p2 = __float2bfloat162_rn(s_alpha_p);
    ds_alpha_p2 = __float2bfloat162_rn(ds_alpha_p);
    ds_alpha_n2 = __float2bfloat162_rn(ds_alpha_n);
    s_alpa_n2 = __float2bfloat162_rn(s_alpha_n);
    beta2 = __float2bfloat162_rn(beta);
    eps2 = __float2bfloat162_rn(eps);
  } else if constexpr (std::is_same<native_t, __half>::value) {
    s_alpa_p2 = __float2half2_rn(s_alpha_p);
    s_alpa_n2 = __float2half2_rn(s_alpha_n);
    ds_alpha_p2 = __float2half2_rn(ds_alpha_p);
    ds_alpha_n2 = __float2half2_rn(ds_alpha_n);
    beta2 = __float2half2_rn(beta);
    eps2 = __float2half2_rn(eps);
  }

  reduction_type thread_dalpha_p = reduction_type(0.0);
  reduction_type thread_dalpha_n = reduction_type(0.0);

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < total_elements / (packed_size * nelements_per_thread);
       i += stride) {

    int base_idx = i * (packed_size * nelements_per_thread);

    if constexpr (std::is_same<native_t, __nv_bfloat16>::value ||
                  std::is_same<native_t, __half>::value) {
#pragma unroll
      for (int j = 0; j < nelements_per_thread; ++j) {
        vec_t x_v =
            reinterpret_cast<const vec_t &>(x[base_idx + j * packed_size]);
        vec_t grad_output_v = reinterpret_cast<const vec_t &>(
            grad_outputs[base_idx + j * packed_size]);

        vec_t dx_v = compute_dx_f16(x_v, grad_output_v, s_alpa_p2, s_alpa_n2,
                                    beta2, eps2);
        vec_t dp_v = compute_dp_f16(x_v, grad_output_v, ds_alpha_p2);
        vec_t dn_v = compute_dn_f16(x_v, grad_output_v, ds_alpha_n2, eps2);

        *reinterpret_cast<vec_t *>(&dx[base_idx + j * packed_size]) = dx_v;

        thread_dalpha_p += float16_sum(dp_v);
        thread_dalpha_n += float16_sum(dn_v);
      }
    } else {
#pragma unroll
      for (int j = 0; j < nelements_per_thread; ++j) {
        vec_t x_v =
            *reinterpret_cast<const vec_t *>(&x[base_idx + j * packed_size]);

        vec_t grad_output_v = reinterpret_cast<const vec_t &>(
            grad_outputs[base_idx + j * packed_size]);

        vec_t dx_v;
        vec_t dalpha_p_v, dalpha_n_v;

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

        *reinterpret_cast<vec_t *>(&dx[base_idx + j * packed_size]) = dx_v;

        thread_dalpha_p +=
            dalpha_p_v.x + dalpha_p_v.y + dalpha_p_v.z + dalpha_p_v.w;
        thread_dalpha_n +=
            dalpha_n_v.x + dalpha_n_v.y + dalpha_n_v.z + dalpha_n_v.w;
      }
    }
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

  TensorOptions options = x.options();

  Tensor dx = torch::empty_like(x);

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
        using Traits = VectorIO<scalar_t>;

        const int loads_per_thread =
            with_vector_loads ? (Traits::packed_size == 2 ? 4 : 1) : 1;

        const int elements_per_thread =
            with_vector_loads ? loads_per_thread * Traits::packed_size : 1;

        const int adjusted_elements = nelements / elements_per_thread;
        const int numBlocks =
            max(1, min(getMaxBlocks(),
                       (adjusted_elements + blockSize - 1) / blockSize));

        if (with_vector_loads) {
          vectorised_xielu_backward_impl<scalar_t, Traits::reduction_t,
                                         (Traits::packed_size == 2 ? 4 : 1)>
              <<<numBlocks, blockSize, 0, stream>>>(
                  x.data_ptr<scalar_t>(), nelements,
                  get_accessor<scalar_t, 1>(alpha_p),
                  get_accessor<scalar_t, 1>(alpha_n),
                  grad_outputs[0].data_ptr<scalar_t>(), (scalar_t)beta,
                  (scalar_t)eps, dx.data_ptr<scalar_t>(),
                  get_accessor<Traits::reduction_t, 1>(dalpha_p),
                  get_accessor<Traits::reduction_t, 1>(dalpha_n));
        } else {
          xielu_backward_impl<scalar_t, Traits::reduction_t>
              <<<numBlocks, blockSize, 0, stream>>>(
                  x.data_ptr<scalar_t>(), nelements,
                  get_accessor<scalar_t, 1>(alpha_p),
                  get_accessor<scalar_t, 1>(alpha_n),
                  grad_outputs[0].data_ptr<scalar_t>(), (scalar_t)beta,
                  (scalar_t)eps, dx.data_ptr<scalar_t>(),
                  get_accessor<Traits::reduction_t, 1>(dalpha_p),
                  get_accessor<Traits::reduction_t, 1>(dalpha_n));
        }
      }));

  torch::Tensor undef;

  POP_RANGE

  return {dx,   dalpha_p.to(x.dtype()), dalpha_n.to(x.dtype()), undef, undef,
          undef};
}
