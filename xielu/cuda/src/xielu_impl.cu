#include "utils.cuh"
#include "xielu_impl.hpp"

#include <iostream>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/script.h>
//#include <torch/nn/functional.h>
#include <math.h>

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

template <typename scalar_t>
__device__ scalar_t softplus(scalar_t x) {
    return x > 20 ? x : log1p(exp(x)); // Numerically stable version of log(1 + exp(x))
}

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

        auto x_e = x[batch_idx][seq_idx][hidden_idx];
        auto s_alpha_p = softplus(alpha_p[hidden_idx]);
        auto s_alpha_n = beta + softplus(alpha_n[hidden_idx]);

        output[batch_idx][seq_idx][hidden_idx] = x_e > 0 ?
            s_alpha_p * x_e * x_e + beta * x_e :
            s_alpha_n * expm1(min(x_e, eps)) - s_alpha_n * x_e + beta * x_e;
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

    // Initialize shared memory
    local_dalpha_p[threadIdx.x] = 0;
    local_dalpha_n[threadIdx.x] = 0;

    if (idx < total_elements) {
        int batch_idx = idx / (seq_len * hidden_dim);
        int residual = idx - batch_idx * seq_len * hidden_dim;
        int seq_idx = residual / hidden_dim;
        int hidden_idx = residual - seq_idx * hidden_dim;

        auto x_e = x[batch_idx][seq_idx][hidden_idx];
        auto grad_output = grad_outputs[batch_idx][seq_idx][hidden_idx];
        auto s_alpha_p = softplus(alpha_p[hidden_idx]);
        auto s_alpha_n = beta + softplus(alpha_n[hidden_idx]);

        if (x_e > 0) {
            dx[batch_idx][seq_idx][hidden_idx] = grad_output * (2 * s_alpha_p * x_e + beta);

            auto alpha_p_e = alpha_p[hidden_idx];
            scalar_t d_s_alpha_p = (alpha_p_e > 20 ? 1 : (exp(alpha_p_e)/log1p(exp(alpha_p_e))));
            local_dalpha_p[threadIdx.x] += grad_output * x_e * x_e * d_s_alpha_p;
            //atomicAdd(&dalpha_p[hidden_idx], grad_output * x_e * x_e * d_s_alpha_p);
        }
        else {
            scalar_t d_expm1_dx = x_e < eps ? exp(x_e) : 0;
            dx[batch_idx][seq_idx][hidden_idx] = grad_output * (s_alpha_n * d_expm1_dx - s_alpha_n + beta);

            auto alpha_n_e = alpha_n[hidden_idx];
            scalar_t d_s_alpha_n = (alpha_n_e > 20 ? 1 : (exp(alpha_n_e)/log1p(exp(alpha_n_e))));
            local_dalpha_n[threadIdx.x] += grad_output * (expm1(min(x_e, eps)) * d_s_alpha_n - x_e * d_s_alpha_n);
            //atomicAdd(&dalpha_n[hidden_idx], grad_output * (expm1(min(x_e, eps)) * d_s_alpha_n - x_e * d_s_alpha_n));
        }
    }

    // Reduce shared memory contributions to global memory
    __syncthreads();

    // Perform block-wise reduction
    if (threadIdx.x == 0) {
        scalar_t sum_dalpha_p = 0;
        scalar_t sum_dalpha_n = 0;

        for (int i = 0; i < blockDim.x; i++) {
            sum_dalpha_p += local_dalpha_p[i];
            sum_dalpha_n += local_dalpha_n[i];
        }

        atomicAdd(&dalpha_p[blockIdx.x], sum_dalpha_p);
        atomicAdd(&dalpha_n[blockIdx.x], sum_dalpha_n);
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

    AT_DISPATCH_FLOATING_TYPES(
        x.scalar_type(), "forward", ([&] {
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

    AT_DISPATCH_FLOATING_TYPES(
        x.scalar_type(), "backward", ([&] {
            // size_t space = 0;
            // void *sptr;
            const int sharedMemSize = 2 * blockSize * sizeof(scalar_t);
            backward_kernel<<<numBlocks, blockSize, sharedMemSize, stream>>>(
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
