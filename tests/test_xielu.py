import pytest
import torch
from torch.autograd import gradcheck
from xielu.ops.wrappers import XIELUPy, XIELUfn, XIELU

import torch.nn.functional as F

NBATCH = 1
NSEQ = 32
HIDDENDIM = 32


@pytest.fixture
def xielu_setup():
    """Fixture to set up XIELU instances and test input."""
    alpha_p_init = 0.8111
    alpha_n_init = 0.8111
    beta = 0.5
    eps = 1e-6

    device = torch.device("cuda")

    input_tensor = torch.randn(
        NBATCH,
        NSEQ,
        HIDDENDIM,
        device=device,
        dtype=torch.float64,
        requires_grad=True,
    )

    xielu_py = XIELUPy(alpha_p_init, alpha_n_init, beta, eps, device, input_tensor.dtype)
    xielu_fn = XIELUfn(alpha_p_init, alpha_n_init, beta, eps, device, input_tensor.dtype)
    xielu = XIELU(alpha_p_init, alpha_n_init, beta, eps, device, input_tensor.dtype)
    xielu_compiled = torch.compile(
        XIELU(alpha_p_init, alpha_n_init, beta, eps, device, input_tensor.dtype)
    )

    return {
        "input": input_tensor,
        "xielu_py": xielu_py,
        "xielu_fn": xielu_fn,
        "xielu": xielu,
        "xielu_compiled": xielu_compiled,
        "device": device,
        "dtype": input_tensor.dtype,
        "alpha_p_init": alpha_p_init,
        "alpha_n_init": alpha_n_init,
        "beta": beta,
        "eps": eps,
    }


def run_forward_comparison(input_tensor, xielu_py, xielu_fn, xielu, xielu_compiled):
    """Helper function to compare forward outputs across implementations."""
    output_py = xielu_py(input_tensor)
    output_fn = xielu_fn(input_tensor)
    output_cuda = xielu(input_tensor)
    output_compiled = xielu_compiled(input_tensor)

    torch.testing.assert_close(output_py, output_fn, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(output_py, output_cuda, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(output_py, output_compiled, rtol=1e-6, atol=1e-6)


def run_gradcheck(model, inputs, dtype=torch.float64, eps=1e-3, rtol=1e-4, atol=1e-4):
    """Helper function to run gradcheck on a model."""
    inputs = tuple(inp.clone().detach().to(dtype).requires_grad_(True) for inp in inputs)
    gradcheck(model, inputs, eps=eps, rtol=rtol, atol=atol, fast_mode=False)


def test_forward(xielu_setup):
    """Test forward pass across all implementations."""
    run_forward_comparison(
        xielu_setup["input"],
        xielu_setup["xielu_py"],
        xielu_setup["xielu_fn"],
        xielu_setup["xielu"],
        xielu_setup["xielu_compiled"],
    )


def test_gradients(xielu_setup):
    """Test gradients for all implementations."""
    run_gradcheck(xielu_setup["xielu"], (xielu_setup["input"],))
    run_gradcheck(xielu_setup["xielu_py"], (xielu_setup["input"],))
    run_gradcheck(xielu_setup["xielu_fn"], (xielu_setup["input"],))
    run_gradcheck(xielu_setup["xielu_compiled"], (xielu_setup["input"],))


def test_alpha_p_alpha_n_gradients(xielu_setup):
    """Test gradients with respect to alpha_p and alpha_n parameters."""
    input_tensor = xielu_setup["input"]
    xielu_py = xielu_setup["xielu_py"]
    xielu_fn = xielu_setup["xielu_fn"]
    xielu = xielu_setup["xielu"]

    run_gradcheck(
        xielu_py.forward_inference,
        (input_tensor, xielu_py.alpha_p, xielu_py.alpha_n),
    )
    run_gradcheck(
        xielu_fn.forward_inference,
        (input_tensor, xielu.alpha_p, xielu.alpha_n),
    )
    run_gradcheck(
        xielu.forward_inference,
        (input_tensor, xielu.alpha_p, xielu.alpha_n),
    )


@pytest.mark.parametrize(
    "case,value",
    [
        ("zero", 0.0),
        ("large", 1e10),
        ("small", 1e-10),
        ("negative", -1.0),
    ],
)
def test_edge_cases(xielu_setup, case, value):
    """Test edge cases with different input values."""
    input_tensor = torch.full_like(
        xielu_setup["input"],
        value,
        dtype=torch.float64,
        requires_grad=True,
    )

    run_forward_comparison(
        input_tensor,
        xielu_setup["xielu_py"],
        xielu_setup["xielu_fn"],
        xielu_setup["xielu"],
        xielu_setup["xielu_compiled"],
    )


def test_xielu_detailed(xielu_setup):
    """Test XIELU class with detailed element-wise comparison"""
    test_input = xielu_setup["input"]
    xielu_py = xielu_setup["xielu_py"]
    xielu = xielu_setup["xielu"]
    xielu_compiled = xielu_setup["xielu_compiled"]

    print("\nTesting XIELU implementations...")
    print(f"Input shape: {test_input.shape}")
    print(f"Device: {xielu_setup['device']}")
    print(f"Input sample values:\n{test_input}")
    print()

    # Test regular XIELU
    print("1. Testing regular XIELU...")
    output_py = xielu_py(test_input)
    output_xielu = xielu(test_input)

    print(f"Python output sample:\n{output_py[0, :2, :2]}")
    print(f"XIELU output sample:\n{output_xielu[0, :2, :2]}")

    diff = torch.abs(output_py - output_xielu)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)

    print(f"Max absolute difference: {max_diff.item():.2e}")
    print(f"Mean absolute difference: {mean_diff.item():.2e}")

    # Show locations of largest differences
    if max_diff > 1e-6:
        max_indices = torch.where(diff == max_diff)
        print(f"Max diff locations: {list(zip(*[idx.tolist() for idx in max_indices]))[:5]}")
        for i, (b, h, w) in enumerate(zip(*max_indices)):
            if i >= 3:
                break  # Only show first 3
            print(f"  [{b},{h},{w}]: py={output_py[b,h,w]:.6f}, xielu={output_xielu[b,h,w]:.6f}, diff={diff[b,h,w]:.6f}")

    torch.testing.assert_close(output_py, output_xielu, rtol=1e-6, atol=1e-6)
    print("Regular XIELU test passed")

    # Test compiled XIELU
    print("\n2. Testing compiled XIELU...")
    output_compiled = xielu_compiled(test_input)

    print(f"Compiled output sample:\n{output_compiled[0, :2, :2]}")

    diff = torch.abs(output_py - output_compiled)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)

    print(f"Max absolute difference: {max_diff.item():.2e}")
    print(f"Mean absolute difference: {mean_diff.item():.2e}")

    # Show locations of largest differences
    if max_diff > 1e-6:
        max_indices = torch.where(diff == max_diff)
        print(f"Max diff locations: {list(zip(*[idx.tolist() for idx in max_indices]))[:5]}")
        for i, (b, h, w) in enumerate(zip(*max_indices)):
            if i >= 3:
                break
            print(f"  [{b},{h},{w}]: py={output_py[b,h,w]:.6f}, compiled={output_compiled[b,h,w]:.6f}, diff={diff[b,h,w]:.6f}")

    torch.testing.assert_close(output_py, output_compiled, rtol=1e-6, atol=1e-6)
    print("Compiled XIELU test passed")

    # Test gradient computation with detailed comparison
    print("\n3. Testing gradients...")

    # Create fresh inputs for gradient testing
    grad_input_py = test_input.clone().detach().requires_grad_(True)
    grad_input_xielu = test_input.clone().detach().requires_grad_(True)
    grad_input_compiled = test_input.clone().detach().requires_grad_(True)

    print(f"Gradient test input shape: {grad_input_py.shape}")
    print(f"Gradient test input sample:\n{grad_input_py[0, :2, :2]}")

    # Forward pass and compute gradients
    output_py_grad = xielu_py(grad_input_py)
    output_xielu_grad = xielu(grad_input_xielu)
    output_compiled_grad = xielu_compiled(grad_input_compiled)

    # Create a dummy loss (sum of outputs)
    loss_py = output_py_grad.sum()
    loss_xielu = output_xielu_grad.sum()
    loss_compiled = output_compiled_grad.sum()

    # Backward pass
    loss_py.backward()
    loss_xielu.backward()
    loss_compiled.backward()

    # Get gradients
    grad_py = grad_input_py.grad
    grad_xielu = grad_input_xielu.grad
    grad_compiled = grad_input_compiled.grad

    print(f"\nPython gradients sample:\n{grad_py[0, :2, :2]}")
    print(f"XIELU gradients sample:\n{grad_xielu[0, :2, :2]}")
    print(f"Compiled gradients sample:\n{grad_compiled[0, :2, :2]}")

    # Compare XIELU vs Python gradients
    grad_diff_xielu = torch.abs(grad_py - grad_xielu)
    max_grad_diff_xielu = torch.max(grad_diff_xielu)
    mean_grad_diff_xielu = torch.mean(grad_diff_xielu)

    print(f"\nXIELU vs Python gradients:")
    print(f"Max absolute gradient difference: {max_grad_diff_xielu.item():.2e}")
    print(f"Mean absolute gradient difference: {mean_grad_diff_xielu.item():.2e}")

    if max_grad_diff_xielu > 1e-6:
        max_indices = torch.where(grad_diff_xielu == max_grad_diff_xielu)
        print(f"Max gradient diff locations: {list(zip(*[idx.tolist() for idx in max_indices]))[:3]}")
        for i, (b, h, w) in enumerate(zip(*max_indices)):
            if i >= 3:
                break
            print(f"  [{b},{h},{w}]: py_grad={grad_py[b,h,w]:.6f}, xielu_grad={grad_xielu[b,h,w]:.6f}, diff={grad_diff_xielu[b,h,w]:.6f}")

    # Compare Compiled vs Python gradients
    grad_diff_compiled = torch.abs(grad_py - grad_compiled)
    max_grad_diff_compiled = torch.max(grad_diff_compiled)
    mean_grad_diff_compiled = torch.mean(grad_diff_compiled)

    print(f"\nCompiled vs Python gradients:")
    print(f"Max absolute gradient difference: {max_grad_diff_compiled.item():.2e}")
    print(f"Mean absolute gradient difference: {mean_grad_diff_compiled.item():.2e}")

    if max_grad_diff_compiled > 1e-6:
        max_indices = torch.where(grad_diff_compiled == max_grad_diff_compiled)
        print(f"Max gradient diff locations: {list(zip(*[idx.tolist() for idx in max_indices]))[:3]}")
        for i, (b, h, w) in enumerate(zip(*max_indices)):
            if i >= 3:
                break
            print(f"  [{b},{h},{w}]: py_grad={grad_py[b,h,w]:.6f}, compiled_grad={grad_compiled[b,h,w]:.6f}, diff={grad_diff_compiled[b,h,w]:.6f}")

    # Run formal gradcheck
    print(f"\n4. Running formal gradcheck ...")
    tiny_input = test_input.clone().detach().requires_grad_(True)

    def xielu_wrapper(x):
        return xielu(x)

    def xielu_compiled_wrapper(x):
        return xielu_compiled(x)

    gradcheck(xielu_wrapper, (tiny_input,), eps=1e-3, rtol=1e-4, atol=1e-4)
    gradcheck(xielu_compiled_wrapper, (tiny_input,), eps=1e-3, rtol=1e-4, atol=1e-4)

    # Tolerance checks for manual gradients
    torch.testing.assert_close(grad_py, grad_xielu, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(grad_py, grad_compiled, rtol=1e-4, atol=1e-4)
    print("All gradient tests passed")

    print("\nAll detailed tests passed!")


@pytest.mark.parametrize("with_vector_loads", [True, False])
def test_reduced_precision(xielu_setup, with_vector_loads):
    """Test XIELU with reduced precision (bfloat16) and larger hidden dimension"""
    alpha_p_init = xielu_setup["alpha_p_init"]
    alpha_n_init = xielu_setup["alpha_n_init"]
    beta = xielu_setup["beta"]
    eps = xielu_setup["eps"]
    device = xielu_setup["device"]

    # Use larger hidden dimension and reduced precision
    ref_dtype = torch.float32
    dtype = torch.bfloat16
    hiddendim_large = 1024

    input_ref = torch.randn(
        NBATCH,
        NSEQ,
        hiddendim_large,
        device=device,
        dtype=ref_dtype,
        requires_grad=True,
    )

    input_reduced = input_ref.clone().detach().to(dtype).requires_grad_(True)

    xielu_py = XIELUPy(alpha_p_init, alpha_n_init, beta, eps, device, input_ref.dtype)
    xielu = XIELU(
        alpha_p_init,
        alpha_n_init,
        beta,
        eps,
        device,
        input_reduced.dtype,
        with_vector_loads=with_vector_loads,
    )

    print(f"\nTesting reduced precision with with_vector_loads={with_vector_loads}")

    # Forward pass
    out_cuda = xielu.forward(input_reduced)
    out_py = xielu_py.forward(input_ref)

    print(f"CUDA output (bfloat16):\n{out_cuda[:1, :2, :5]}")
    print(f"Python output (float32):\n{out_py[:1, :2, :5]}")
    mean_output_error = (out_cuda.float() - out_py).abs().mean()
    print(f"Mean output error: {mean_output_error.item():.6e}")

    # Backward pass
    grad_output_cuda = torch.ones_like(out_cuda)
    out_cuda.backward(grad_output_cuda, retain_graph=True)

    grad_output_py = torch.ones_like(out_py)
    out_py.backward(grad_output_py, retain_graph=True)

    # Check input gradient
    mean_input_grad_error = (input_reduced.grad.float() - input_ref.grad).abs().mean()
    print(f"Mean input.grad error: {mean_input_grad_error.item():.6e}")

    # Check alpha_p gradient
    print(f"alpha_p.grad (py): {xielu_py.alpha_p.grad}")
    print(f"alpha_p.grad (cuda): {xielu.alpha_p.grad}")
    mean_alpha_p_error = (xielu_py.alpha_p.grad - xielu.alpha_p.grad.float()).abs().mean()
    print(f"Mean alpha_p.grad error: {mean_alpha_p_error.item():.6e}")

    # Check alpha_n gradient
    print(f"alpha_n.grad (py): {xielu_py.alpha_n.grad}")
    print(f"alpha_n.grad (cuda): {xielu.alpha_n.grad}")
    mean_alpha_n_error = (xielu_py.alpha_n.grad - xielu.alpha_n.grad.float()).abs().mean()
    print(f"Mean alpha_n.grad error: {mean_alpha_n_error.item():.6e}")

    # Assert tolerances are reasonable for bfloat16
    # bfloat16 has less precision, so we use looser tolerances
    assert mean_output_error < 1e-2, f"Output error too large: {mean_output_error}"
    assert mean_input_grad_error < 1e-2, f"Input gradient error too large: {mean_input_grad_error}"
    assert mean_alpha_p_error < 1e-2, f"alpha_p gradient error too large: {mean_alpha_p_error}"
    assert mean_alpha_n_error < 1e-2, f"alpha_n gradient error too large: {mean_alpha_n_error}"

    print("Reduced precision test passed!")
