#!/usr/bin/env python3
"""Test script to compare XIELU with and without stochastic rounding for reduced precision types."""

import torch
from xielu.ops.wrappers import XIELUPy, XIELU

# Test configuration
NBATCH = 1
NSEQ = 32
HIDDENDIM = 1024
NUM_ITERATIONS = 10

alpha_p_init = 0.8111
alpha_n_init = 0.8111
beta = 0.5
eps = 1e-5

device = torch.device("cuda")

def test_stochastic_rounding(dtype):
    """Test stochastic rounding for a specific dtype."""
    print(f"\n{'='*80}")
    print(f"Testing dtype: {dtype}")
    print(f"{'='*80}")

    # Always use F64 as reference
    ref_dtype = torch.float64

    # Create F64 reference input
    input_ref = torch.randn(
        NBATCH,
        NSEQ,
        HIDDENDIM,
        device=device,
        dtype=ref_dtype,
        requires_grad=True,
    )

    # Convert to test dtype
    input_test = input_ref.clone().detach().to(dtype).requires_grad_(True)

    # F64 reference implementation
    xielu_ref = XIELUPy(alpha_p_init, alpha_n_init, beta, eps, device, ref_dtype)

    # Test dtype implementations
    xielu_deterministic = XIELU(
        alpha_p_init, alpha_n_init, beta, eps, device, dtype,
        with_vector_loads=True, use_stochastic_rounding=False
    )

    xielu_stochastic = XIELU(
        alpha_p_init, alpha_n_init, beta, eps, device, dtype,
        with_vector_loads=True, use_stochastic_rounding=True
    )

    # Reference forward
    out_ref = xielu_ref.forward(input_ref)

    # Deterministic forward (single pass)
    input_det = input_test.clone().detach().requires_grad_(True)
    out_det = xielu_deterministic.forward(input_det)

    # Stochastic forward (multiple passes to see variance)
    stochastic_outputs = []
    stochastic_inputs = []
    for _ in range(NUM_ITERATIONS):
        input_stoch = input_test.clone().detach().requires_grad_(True)
        stochastic_inputs.append(input_stoch)
        out_stoch = xielu_stochastic.forward(input_stoch)
        stochastic_outputs.append(out_stoch)

    # Compare outputs
    print(f"\n--- Forward Pass Analysis ---")

    # Deterministic vs Reference
    det_error = (out_det.double() - out_ref).abs()
    print(f"\nDeterministic (no SR):")
    print(f"  Mean error vs F64 ref: {det_error.mean().item():.6e}")
    print(f"  Max error vs F64 ref:  {det_error.max().item():.6e}")
    print(f"  Std error vs F64 ref:  {det_error.std().item():.6e}")

    # Stochastic vs Reference (averaged)
    stoch_errors = []
    for out_stoch in stochastic_outputs:
        error = (out_stoch.double() - out_ref).abs()
        stoch_errors.append(error)

    stoch_error_mean = torch.stack([e.mean() for e in stoch_errors])
    stoch_error_max = torch.stack([e.max() for e in stoch_errors])

    print(f"\nStochastic (with SR) - {NUM_ITERATIONS} runs:")
    print(f"  Mean error vs F64 ref (avg): {stoch_error_mean.mean().item():.6e} ± {stoch_error_mean.std().item():.6e}")
    print(f"  Max error vs F64 ref (avg):  {stoch_error_max.mean().item():.6e} ± {stoch_error_max.std().item():.6e}")

    # Variance between stochastic runs
    stoch_variance = torch.stack(stochastic_outputs).std(dim=0)
    print(f"\nVariance between stochastic runs:")
    print(f"  Mean variance: {stoch_variance.mean().item():.6e}")
    print(f"  Max variance:  {stoch_variance.max().item():.6e}")

    # Backward pass analysis
    print(f"\n--- Backward Pass Analysis ---")

    # Deterministic backward
    loss_det = out_det.sum()
    loss_det.backward()

    # Reference backward
    loss_ref = out_ref.sum()
    loss_ref.backward()

    # Stochastic backward (multiple runs)
    stochastic_grads = []
    for input_stoch, out_stoch in zip(stochastic_inputs, stochastic_outputs):
        loss_stoch = out_stoch.sum()
        loss_stoch.backward()
        stochastic_grads.append(input_stoch.grad.clone())

    # Compare gradients
    det_grad_error = (input_det.grad.double() - input_ref.grad).abs()
    print(f"\nDeterministic (no SR):")
    print(f"  Mean input grad error: {det_grad_error.mean().item():.6e}")
    print(f"  Max input grad error:  {det_grad_error.max().item():.6e}")

    print(f"  alpha_p grad error: {abs(xielu_deterministic.alpha_p.grad.item() - xielu_ref.alpha_p.grad.item()):.6e}")
    print(f"  alpha_n grad error: {abs(xielu_deterministic.alpha_n.grad.item() - xielu_ref.alpha_n.grad.item()):.6e}")

    # Stochastic gradients
    stoch_grad_errors = []
    for grad_stoch in stochastic_grads:
        error = (grad_stoch.double() - input_ref.grad).abs()
        stoch_grad_errors.append(error)

    stoch_grad_error_mean = torch.stack([e.mean() for e in stoch_grad_errors])
    stoch_grad_error_max = torch.stack([e.max() for e in stoch_grad_errors])

    print(f"\nStochastic (with SR) - {NUM_ITERATIONS} runs:")
    print(f"  Mean input grad error (avg): {stoch_grad_error_mean.mean().item():.6e} ± {stoch_grad_error_mean.std().item():.6e}")
    print(f"  Max input grad error (avg):  {stoch_grad_error_max.mean().item():.6e} ± {stoch_grad_error_max.std().item():.6e}")

    # Gradient variance
    grad_variance = torch.stack(stochastic_grads).std(dim=0)
    print(f"\nGradient variance between stochastic runs:")
    print(f"  Mean variance: {grad_variance.mean().item():.6e}")
    print(f"  Max variance:  {grad_variance.max().item():.6e}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    print("Testing Stochastic Rounding vs Deterministic Rounding")
    print("=" * 80)

    # Test bfloat16
    test_stochastic_rounding(torch.bfloat16)

    # Test float16
    test_stochastic_rounding(torch.float16)

    print("\nTesting complete!")
