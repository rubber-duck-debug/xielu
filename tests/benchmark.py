import torch
import torch.utils.benchmark as benchmark
from xielu.ops.wrappers import XIELUPy, XIELU, XIELUfn

def benchmark_model(model, input_tensor, label):
    """Benchmark forward and backward passes of a model."""
    results = []
    # Forward pass benchmark
    t_forward = benchmark.Timer(
        stmt="model(input_tensor)",
        globals={"model": model, "input_tensor": input_tensor},
        num_threads=1,
        label=label,
        sub_label="Forward",
        description="Forward Pass",
    )
    results.append(t_forward.timeit(100))  # Run 100 iterations

    # Backward pass benchmark
    input_tensor.requires_grad_(True)
    output = model(input_tensor)
    grad_output = torch.ones_like(output)

    t_backward = benchmark.Timer(
        stmt="output.backward(grad_output, retain_graph=True)",
        globals={"output": output, "grad_output": grad_output},
        num_threads=1,
        label=label,
        sub_label="Backward",
        description="Backward Pass",
    )
    results.append(t_backward.timeit(100))  # Run 100 iterations

    return results

def run_benchmarks():
    """Run benchmarks for Python, CUDA, and torch.compile implementations."""
    device = torch.device("cuda")

    # Set batch size, sequence length, and hidden dimensions
    NBATCH, NSEQ, HIDDENDIM = 5, 4096, 8192

    # Create input tensor
    input_tensor = torch.randn(NBATCH, NSEQ, HIDDENDIM, device=device, dtype=torch.float32)

    # Initialize models
    xielu_py = torch.compile(XIELUPy(0.8, 0.8, 0.5, 1e-6)).to(device)
    #xielu_fn = XIELUfn(0.8, 0.8, 0.5, 1e-6).to(device)
    xielu_cuda = torch.compile(XIELU(0.8, 0.8, 0.5, 1e-6)).to(device)

    # Run benchmarks
    results = []
    results += benchmark_model(xielu_py, input_tensor, "XIELU-Python")
    results += benchmark_model(xielu_cuda, input_tensor, "XIELU-CUDA")

    # Print results
    for r in results:
        print(r)

if __name__ == "__main__":
    run_benchmarks()
