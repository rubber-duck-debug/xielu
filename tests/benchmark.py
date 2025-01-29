import torch
import torch.utils.benchmark as benchmark
from xielu.ops.wrappers import XIELUPy, XIELU, XIELUfn

WARMUP_ITERS = 10
TIMING_ITERS = 100
INPUT_SIZES = [
    (4, 16, 128),
    (8, 32, 256),
    (16, 32, 128),
    (16, 64, 512),
    (32, 128, 1024),
]

def benchmark_model(model, input_tensor, label):
    results = []

    # Forward pass benchmark
    for _ in range(WARMUP_ITERS):
        _ = model(input_tensor)

    t_forward = benchmark.Timer(
        stmt="model(input_tensor)",
        globals={"model": model, "input_tensor": input_tensor},
        num_threads=1,
        label=label,
        sub_label="Forward",
        description="Forward Pass",
    )
    results.append(t_forward.timeit(TIMING_ITERS))

    # Backward pass benchmark
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    grad_output = torch.ones_like(output)

    for _ in range(WARMUP_ITERS):
        output.backward(grad_output, retain_graph=True)

    t_backward = benchmark.Timer(
        stmt="output.backward(grad_output, retain_graph=True)",
        globals={"output": output, "grad_output": grad_output},
        num_threads=1,
        label=label,
        sub_label="Backward",
        description="Backward Pass",
    )
    results.append(t_backward.timeit(TIMING_ITERS))

    return results

def run_benchmarks():
    device = torch.device("cuda")
    results = []

    for (NBATCH, NSEQ, HIDDENDIM) in INPUT_SIZES:
        print(f"\nRunning Benchmarks for Input Shape: ({NBATCH}, {NSEQ}, {HIDDENDIM})")

        # Create input tensor
        input_tensor = torch.randn(NBATCH, NSEQ, HIDDENDIM, device=device, dtype=torch.float32)

        # Initialize models
        xielu_py = XIELUPy(0.8, 0.8, 0.5, 1e-6).to(device)
        #xielu_fn = XIELUfn(0.8, 0.8, 0.5, 1e-6).to(device)
        xielu_cuda = XIELU(0.8, 0.8, 0.5, 1e-6).to(device)

        # Run benchmarks
        results += benchmark_model(xielu_py, input_tensor, f"XIELU-Python ({NBATCH}, {NSEQ}, {HIDDENDIM})")
        #results += benchmark_model(xielu_fn, input_tensor, f"XIELU-Python-fn ({NBATCH}, {NSEQ}, {HIDDENDIM})")
        results += benchmark_model(xielu_cuda, input_tensor, f"XIELU-CUDA ({NBATCH}, {NSEQ}, {HIDDENDIM})")

    # Print results
    print("\nBenchmark Results:")
    for r in results:
        print(r)

if __name__ == "__main__":
    run_benchmarks()
