import torch
import torch.utils.benchmark as benchmark
from xielu.ops.wrappers import XIELUPy, XIELU

WARMUP_ITERS = 10
#TIMING_ITERS = 100
INPUT_SIZES = [
    ( 4,    16,  128),
    ( 8,    32,  256),
    ( 16,   32,  128),
    ( 16,   64,  512),
    ( 32,  128, 1024),
    (  5, 4096, 8192),
    ( 50, 4096, 8192),
]

def benchmark_model(model, input_tensor, label):
    results = {}

    torch.cuda.synchronize()
    # Forward pass benchmark
    for _ in range(WARMUP_ITERS):
        _ = model(input_tensor)
    torch.cuda.synchronize()
    t_forward = benchmark.Timer(
        stmt="model(input_tensor)",
        globals={"model": model, "input_tensor": input_tensor},
        num_threads=1,
        label=label,
        sub_label="Forward",
        description="Forward Pass",
    )
    #results["forward"] = t_forward.timeit(TIMING_ITERS)
    results["forward"] = t_forward.blocked_autorange(min_run_time=1.0)
    torch.cuda.synchronize()

    # Backward pass benchmark
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    grad_output = torch.ones_like(output)
    for _ in range(WARMUP_ITERS):
        output.backward(grad_output, retain_graph=True)
    torch.cuda.synchronize()
    t_backward = benchmark.Timer(
        stmt="output.backward(grad_output, retain_graph=True)",
        globals={"output": output, "grad_output": grad_output},
        num_threads=1,
        label=label,
        sub_label="Backward",
        description="Backward Pass",
    )
    #results["backward"] = t_backward.timeit(TIMING_ITERS)
    results["backward"] = t_backward.blocked_autorange(min_run_time=1.0)
    torch.cuda.synchronize()

    return label, results

def format_results(results):
    """Format and print benchmark results in a clean table."""
    print("\nBenchmark Results:")
    print(f"{'Model':<30}{'Batch':<10}{'SeqLen':<10}{'HiddenDim':<15}{'Fwd (ms)':<15}{'Bwd (ms)':<15}")
    print("=" * 90)

    prev_batch, prev_seq_len, prev_hidden_dim = None, None, None
    for (label, result, batch, seq_len, hidden_dim) in results:
        if (prev_batch, prev_seq_len, prev_hidden_dim) != (batch, seq_len, hidden_dim):
            print("")  # Newline for better readability
            prev_batch, prev_seq_len, prev_hidden_dim = batch, seq_len, hidden_dim
        fwd_time = f"{result['forward'].mean * 1000:.2f}"
        bwd_time = f"{result['backward'].mean * 1000:.2f}"
        print(f"{label:<30}{batch:>10}{seq_len:>10}{hidden_dim:>15}{fwd_time:>15}{bwd_time:>15}")


def run_benchmarks():
    device = torch.device("cuda")
    results = []

    for (NBATCH, NSEQ, HIDDENDIM) in INPUT_SIZES:
        print(f"Running Benchmarks for Input Shape: ({NBATCH}, {NSEQ}, {HIDDENDIM})...")

        # Create input tensor
        input_tensor = torch.randn(NBATCH, NSEQ, HIDDENDIM, device=device, dtype=torch.float32)

        # Initialize models
        xielu_py = torch.compile(XIELUPy(0.8, 0.8, 0.5, 1e-6)).to(device)
        xielu_cuda = torch.compile(XIELU(0.8, 0.8, 0.5, 1e-6)).to(device)

        # Run benchmarks
        results.append((*benchmark_model(xielu_py, input_tensor.clone(), "XIELU-Python"), NBATCH, NSEQ, HIDDENDIM))
        results.append((*benchmark_model(xielu_cuda, input_tensor.clone(), "XIELU-Cuda"), NBATCH, NSEQ, HIDDENDIM))

    format_results(results)

if __name__ == "__main__":
    run_benchmarks()
