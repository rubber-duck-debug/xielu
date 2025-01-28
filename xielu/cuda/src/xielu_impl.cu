#include "utils.cuh"
#include "xielu_impl.hpp"

#include <iostream>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/Atomic.cuh>
#include <torch/script.h>
#include <math.h>
//#include <cuda_bf16.h>
//#include <cuda_fp16.h>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

using torch::Tensor;
using torch::TensorOptions;

#define CHECK_RESULT(result)                                       \
    if (result != cudaSuccess) {                                   \
        throw runtime_error(string("Encountered error ") +         \
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

template<typename T>
struct softplus {
    static __device__ T f(T x) {
        return
            x > T(20.0) ? x : (
                x < T(-20.0) ? T(0.0) :
                    log1p(exp(x)));
    }

    static __device__ T df(T x) {
        return
            x > T(20.0) ? T(1.0) : (
                x < T(-20.0) ? T(0.0) :
                    //sigmoid(x));
                    T(1.0)/(T(1.0)+exp(-x)));
    }
};

template<>
struct softplus<c10::Half> {
    static __device__ c10::Half f(c10::Half x) {
        return static_cast<c10::Half>(softplus<float>::f(static_cast<float>(x))); }
    static __device__ c10::Half df(c10::Half x) {
        return static_cast<c10::Half>(softplus<float>::df(static_cast<float>(x))); }
};

template<>
struct softplus<c10::BFloat16> {
    static __device__ c10::BFloat16 f(c10::BFloat16 x) {
        return static_cast<c10::BFloat16>(softplus<float>::f(static_cast<float>(x))); }
    static __device__ c10::BFloat16 df(c10::BFloat16 x) {
        return static_cast<c10::BFloat16>(softplus<float>::df(static_cast<float>(x))); }
};

template <typename scalar_t>
__global__ void forward_kernel(const Accessor<scalar_t, 3> x,
                               const Accessor<scalar_t, 1> alpha_p,
                               const Accessor<scalar_t, 1> alpha_n,
                               const scalar_t beta,
                               const scalar_t eps,
                               Accessor<scalar_t, 3> output) {

    const int batch_size = x.size(0);
    const int seq_len = x.size(1);
    const int hidden_dim = x.size(2);
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements) {
        int batch_idx = idx / (seq_len * hidden_dim);
        int residual = idx - batch_idx * seq_len * hidden_dim;
        int seq_idx = residual / hidden_dim;
        int hidden_idx = residual - seq_idx * hidden_dim;

        // implementation of
        // alpha_p = F.softplus(self.alpha_p)
        // alpha_n = self.beta + F.softplus(self.alpha_n)
        // return torch.where(x > 0,
        //     alpha_p * x * x + self.beta * x,
        //     alpha_n * torch.expm1(torch.min(x, self.eps)) - alpha_n * x + self.beta * x)

        using sp = softplus<scalar_t>;
        scalar_t x_e = x[batch_idx][seq_idx][hidden_idx];
        scalar_t s_alpha_p = sp::f(alpha_p[0]);
        scalar_t s_alpha_n = sp::f(alpha_n[0]);

        if (static_cast<float>(x_e) > 0.0f) {
            output[batch_idx][seq_idx][hidden_idx] = sp::f(alpha_p[0]) * x_e * x_e + beta * x_e;
        }
        else {
            output[batch_idx][seq_idx][hidden_idx] = (beta + s_alpha_n) * expm1(min(x_e, eps)) - s_alpha_n * x_e;
        }
    }
}

template <typename scalar_t>
__global__ void backward_kernel(const Accessor<scalar_t, 3> x,
                                const Accessor<scalar_t, 1> alpha_p,
                                const Accessor<scalar_t, 1> alpha_n,
                                const Accessor<scalar_t, 3> grad_outputs,
                                const scalar_t beta,
                                const scalar_t eps,
                                Accessor<scalar_t, 3> dx,
                                Accessor<scalar_t, 1> dalpha_p,
                                Accessor<scalar_t, 1> dalpha_n) {

    const int batch_size = x.size(0);
    const int seq_len = x.size(1);
    const int hidden_dim = x.size(2);
    const int total_elements = batch_size * seq_len * hidden_dim;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory to accumulate contributions to dalpha_p and dalpha_n
    __shared__ scalar_t local_dalpha_p[128];
    __shared__ scalar_t local_dalpha_n[128];

    if (idx < total_elements) {
        int batch_idx = idx / (seq_len * hidden_dim);
        int residual = idx - batch_idx * seq_len * hidden_dim;
        int seq_idx = residual / hidden_dim;
        int hidden_idx = residual - seq_idx * hidden_dim;

        using sp = softplus<scalar_t>;
        scalar_t x_e = x[batch_idx][seq_idx][hidden_idx];
        scalar_t grad_output = grad_outputs[batch_idx][seq_idx][hidden_idx];

        if (static_cast<float>(x_e) > 0.0f) {
            dx[batch_idx][seq_idx][hidden_idx] = grad_output * 2 * sp::f(alpha_p[0]) * x_e + beta;
            local_dalpha_p[threadIdx.x] = grad_output * sp::df(alpha_p[0]) * x_e * x_e;
        }
        else {
            if (static_cast<float>(x_e) < static_cast<float>(eps)) {
                dx[batch_idx][seq_idx][hidden_idx] = grad_output * (
                    (beta + sp::f(alpha_n[0])) * expm1(x_e) - sp::f(alpha_n[0]));
            }
            else {
                dx[batch_idx][seq_idx][hidden_idx] = grad_output * (-sp::f(alpha_n[0]));
            }
            local_dalpha_n[threadIdx.x] = grad_output * sp::df(alpha_n[0]) * (expm1(min(x_e, eps)) - x_e);
        }
    }

    // Reduce shared memory contributions to global memory
    __syncthreads();

    // Perform block-wise reduction
    if (threadIdx.x == 0) {
        scalar_t sum_dalpha_p = scalar_t(0.0);
        scalar_t sum_dalpha_n = scalar_t(0.0);

        for (int i = 0; i < blockDim.x; i++) {
            sum_dalpha_p += local_dalpha_p[i];
            sum_dalpha_n += local_dalpha_n[i];
        }

        gpuAtomicAdd(&dalpha_p[blockIdx.x], sum_dalpha_p);
        gpuAtomicAdd(&dalpha_n[blockIdx.x], sum_dalpha_n);
    }
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
            ((batch_size * seq_len * hidden_dim) + blockSize - 1) / blockSize));

    TensorOptions options = x.options();
    Tensor output = torch::empty_like(x);

    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    const c10::cuda::CUDAStreamGuard guard(stream);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "forward", ([&] {
            // size_t space = 0;
            // void *sptr;
            forward_kernel<<<numBlocks, blockSize, 0, stream>>>(
                get_accessor<scalar_t, 3>(x),
                get_accessor<scalar_t, 1>(alpha_p),
                get_accessor<scalar_t, 1>(alpha_n),
                (scalar_t)beta,
                (scalar_t)eps,
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

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, x.scalar_type(), "backward", ([&] {
            // size_t space = 0;
            // void *sptr;
            backward_kernel<<<numBlocks, blockSize, 0, stream>>>(
                get_accessor<scalar_t, 3>(x),
                get_accessor<scalar_t, 1>(alpha_p),
                get_accessor<scalar_t, 1>(alpha_n),
                get_accessor<scalar_t, 3>(grad_outputs[0]),
                (scalar_t)beta,
                (scalar_t)eps,
                get_accessor<scalar_t, 3>(dx),
                get_accessor<scalar_t, 1>(dalpha_p),
                get_accessor<scalar_t, 1>(dalpha_n));
    }));

    POP_RANGE

    return {dx, torch::sum(dalpha_p), torch::sum(dalpha_n)};
}
