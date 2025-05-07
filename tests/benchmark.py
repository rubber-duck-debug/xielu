import torch
import torch.utils.benchmark as benchmark
from xielu.ops.wrappers import XIELUPy, XIELU
from time import time

WARMUP_ITERS = 50
RESULT_ITERS = 100

dtype = torch.bfloat16

INPUT_SIZES = [
    (1, 4096, 8192),
    (2, 2048, 8192),
    (5, 4096, 8192),
]


def benchmark_model(model, input_tensor, label):
    results = {}
    # Forward pass benchmark
    torch.cuda.synchronize()
    for _ in range(WARMUP_ITERS):
        _ = model(input_tensor)
    torch.cuda.synchronize()

    start = time()
    for _ in range(RESULT_ITERS):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    end = time()

    results["forward"] = (end - start) / RESULT_ITERS

    # Backward pass benchmark
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    grad_output = torch.ones_like(output)
    torch.cuda.synchronize()
    start = time()
    for _ in range(RESULT_ITERS):
        output.backward(grad_output, retain_graph=True)
    torch.cuda.synchronize()
    end = time()

    results["backward"] = (end - start) / RESULT_ITERS

    return label, results


def format_results(results, info):
    """Format and print benchmark results in a clean table."""
    print(f"\n{info}")
    print(f"{'Model':<14}{'Batch':>10}{'SeqLen':>10}{'HiddenDim':>10}{'Fwd (ms)':>10}{'Bwd (ms)':>10}")
    print("=" * 64)

    prev_batch, prev_seq_len, prev_hidden_dim = None, None, None
    for (label, result, batch, seq_len, hidden_dim) in results:
        if (prev_batch, prev_seq_len, prev_hidden_dim) != (batch, seq_len, hidden_dim):
            print("")  # Newline for better readability
            prev_batch, prev_seq_len, prev_hidden_dim = batch, seq_len, hidden_dim
        fwd_time = f"{result['forward'] * 1000:.2f}"
        bwd_time = f"{result['backward'] * 1000:.2f}"
        print(
            f"{label:<14}{batch:>10}{seq_len:>10}{hidden_dim:>10}{fwd_time:>10}{bwd_time:>10}")


def run_benchmarks():
    device = torch.device("cuda")

    results = []

    xielu_py = torch.compile(
        XIELUPy(0.8, 0.8, 0.5, -1e-6, dtype=dtype)).to(device)
    xielu_cuda = torch.compile(
        XIELU(0.8, 0.8, 0.5, -1e-6, dtype=dtype, with_vector_loads=False)).to(device)

    for (NBATCH, NSEQ, HIDDENDIM) in INPUT_SIZES:
        print(
            f"Running Benchmarks for Input Shape: ({NBATCH}, {NSEQ}, {HIDDENDIM})...")

        # Create input tensor
        input_tensor = torch.randn(
            NBATCH, NSEQ, HIDDENDIM, device=device, dtype=dtype)

        # Initialize models

        # Run benchmarks
        results.append((*benchmark_model(xielu_py, input_tensor.clone(),
                       "XIELU-Python"), NBATCH, NSEQ, HIDDENDIM))
        results.append((*benchmark_model(xielu_cuda, input_tensor.clone(),
                       "XIELU-Cuda"), NBATCH, NSEQ, HIDDENDIM))

    format_results(results, "Benchmark Results without vector loads:")

    xielu_cuda = torch.compile(
        XIELU(0.8, 0.8, 0.5, 1e-6, dtype=dtype, with_vector_loads=True)).to(device)

    results = []

    for (NBATCH, NSEQ, HIDDENDIM) in INPUT_SIZES:
        print(
            f"Running Benchmarks for Input Shape: ({NBATCH}, {NSEQ}, {HIDDENDIM})...")

        # Create input tensor
        input_tensor = torch.randn(
            NBATCH, NSEQ, HIDDENDIM, device=device, dtype=dtype)

        # Initialize models

        # Run benchmarks
        results.append((*benchmark_model(xielu_cuda, input_tensor.clone(),
                       "XIELU-Cuda"), NBATCH, NSEQ, HIDDENDIM))

    format_results(results, "Benchmark Results with vector loads:")


if __name__ == "__main__":
    run_benchmarks()
