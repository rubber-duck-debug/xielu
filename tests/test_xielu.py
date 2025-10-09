import pytest
import torch
from torch.autograd import gradcheck
from xielu.ops.wrappers import XIELUPy, XIELU

NBATCH = 1
NSEQ = 32
HIDDENDIM = 32


def get_tolerances(dtype):
    """Get appropriate tolerances based on dtype.

    Note: grad_atol_alpha is specifically for accumulated scalar gradients (alpha_p, alpha_n)
    which accumulate errors over many elements, so needs higher tolerance than per-element gradients.
    """
    if dtype == torch.float64:
        return {"rtol": 1e-6, "atol": 1e-6, "grad_eps": 1e-3, "grad_rtol": 1e-4, "grad_atol": 1e-4, "grad_atol_alpha": 1e-3, "threshold": 1e-6}
    elif dtype == torch.float32:
        return {"rtol": 1e-5, "atol": 1e-5, "grad_eps": 1e-3, "grad_rtol": 1e-4, "grad_atol": 1e-4, "grad_atol_alpha": 1e-2, "threshold": 1e-5}
    elif dtype == torch.bfloat16:
        return {"rtol": 1e-2, "atol": 1e-2, "grad_eps": 1e-2, "grad_rtol": 1e-2, "grad_atol": 1e-2, "grad_atol_alpha": 50.0, "threshold": 1e-2}
    elif dtype == torch.float16:
        return {"rtol": 1e-3, "atol": 1e-3, "grad_eps": 1e-3, "grad_rtol": 1e-3, "grad_atol": 1e-3, "grad_atol_alpha": 5.0, "threshold": 1e-3}
    else:
        # Default to float32 tolerances
        return {"rtol": 1e-5, "atol": 1e-5, "grad_eps": 1e-3, "grad_rtol": 1e-4, "grad_atol": 1e-4, "grad_atol_alpha": 1e-2, "threshold": 1e-5}


@pytest.fixture
def xielu_setup():
    """Fixture to set up XIELU instances and test input."""
    alpha_p_init = 0.8111
    alpha_n_init = 0.8111
    beta = 0.5
    eps = 1e-5

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
    xielu = XIELU(alpha_p_init, alpha_n_init, beta, eps, device, input_tensor.dtype)

    return {
        "input": input_tensor,
        "xielu_py": xielu_py,
        "xielu": xielu,
        "device": device,
        "dtype": input_tensor.dtype,
        "alpha_p_init": alpha_p_init,
        "alpha_n_init": alpha_n_init,
        "beta": beta,
        "eps": eps,
    }


def run_forward_comparison(input_tensor, xielu_py, xielu, dtype=None):
    """Helper function to compare XIELU forward outputs against XIELUPy reference.

    Note: This function assumes both implementations use the same dtype.
    For cross-dtype comparisons, see test_reduced_precision and test_xielu_detailed.
    """
    if dtype is None:
        dtype = input_tensor.dtype

    tols = get_tolerances(dtype)

    # Compare XIELU against XIELUPy reference
    output_py = xielu_py(input_tensor)
    output_cuda = xielu(input_tensor)

    torch.testing.assert_close(output_cuda, output_py, rtol=tols["rtol"], atol=tols["atol"])


def run_gradcheck(model, inputs, dtype=None):
    """Helper function to run gradcheck on a model."""
    if dtype is None:
        dtype = inputs[0].dtype

    tols = get_tolerances(dtype)

    inputs = tuple(inp.clone().detach().to(dtype).requires_grad_(True) for inp in inputs)
    gradcheck(model, inputs, eps=tols["grad_eps"], rtol=tols["grad_rtol"], atol=tols["grad_atol"], fast_mode=False)


def test_forward(xielu_setup):
    """Test forward pass: XIELU vs XIELUPy reference."""
    run_forward_comparison(
        xielu_setup["input"],
        xielu_setup["xielu_py"],
        xielu_setup["xielu"],
    )


def test_gradients(xielu_setup):
    """Test gradients: XIELU vs XIELUPy reference."""
    run_gradcheck(xielu_setup["xielu"], (xielu_setup["input"],))
    run_gradcheck(xielu_setup["xielu_py"], (xielu_setup["input"],))


def test_alpha_p_alpha_n_gradients(xielu_setup):
    """Test gradients with respect to alpha_p and alpha_n parameters."""
    input_tensor = xielu_setup["input"]
    xielu_py = xielu_setup["xielu_py"]
    xielu = xielu_setup["xielu"]

    run_gradcheck(
        xielu_py.forward_inference,
        (input_tensor, xielu_py.alpha_p, xielu_py.alpha_n),
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
        xielu_setup["xielu"],
    )


def test_xielu_detailed(xielu_setup):
    """Test XIELU class with detailed element-wise comparison against XIELUPy F64 reference"""
    test_input = xielu_setup["input"]
    dtype = xielu_setup["dtype"]
    device = xielu_setup["device"]
    alpha_p_init = xielu_setup["alpha_p_init"]
    alpha_n_init = xielu_setup["alpha_n_init"]
    beta = xielu_setup["beta"]
    eps = xielu_setup["eps"]

    tols = get_tolerances(dtype)

    # Always create F64 reference
    ref_dtype = torch.float64
    input_ref = test_input.clone().detach().to(ref_dtype).requires_grad_(True)
    input_test = test_input.clone().detach().to(dtype).requires_grad_(True)

    # F64 reference implementation (XIELUPy)
    xielu_ref = XIELUPy(alpha_p_init, alpha_n_init, beta, eps, device, ref_dtype)

    # Test dtype implementation (XIELU)
    xielu = XIELU(alpha_p_init, alpha_n_init, beta, eps, device, dtype)

    print("\nTesting XIELU vs XIELUPy reference...")
    print(f"Input shape: {test_input.shape}")
    print(f"Device: {device}")
    print(f"Test dtype: {dtype}, Reference dtype: {ref_dtype}")
    print(f"Tolerances: rtol={tols['rtol']:.2e}, atol={tols['atol']:.2e}")
    print(f"Input sample values:\n{input_test}")
    print()

    # Test forward pass
    print("1. Testing forward pass...")
    output_ref = xielu_ref(input_ref)
    output_xielu = xielu(input_test)

    print(f"Reference output sample (F64):\n{output_ref[0, :2, :2]}")
    print(f"XIELU output sample ({dtype}):\n{output_xielu[0, :2, :2]}")

    # Convert test output to F64 for comparison
    diff = torch.abs(output_ref - output_xielu.double())
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)

    print(f"Max absolute difference: {max_diff.item():.2e}")
    print(f"Mean absolute difference: {mean_diff.item():.2e}")

    # Show locations of largest differences
    if max_diff > tols["threshold"]:
        max_indices = torch.where(diff == max_diff)
        print(f"Max diff locations: {list(zip(*[idx.tolist() for idx in max_indices]))[:5]}")
        for i, (b, h, w) in enumerate(zip(*max_indices)):
            if i >= 3:
                break  # Only show first 3
            print(f"  [{b},{h},{w}]: ref={output_ref[b,h,w]:.6f}, xielu={output_xielu[b,h,w]:.6f}, diff={diff[b,h,w]:.6f}")

    torch.testing.assert_close(output_xielu.double(), output_ref, rtol=tols["rtol"], atol=tols["atol"])
    print("Forward pass test passed")

    # Test gradient computation with detailed comparison
    print("\n2. Testing gradients...")

    # Create fresh inputs for gradient testing (F64 reference and test dtype)
    grad_input_ref = input_ref.clone().detach().requires_grad_(True)
    grad_input_xielu = input_test.clone().detach().requires_grad_(True)

    print(f"Gradient test input shape: {grad_input_ref.shape}")
    print(f"Gradient test input sample:\n{grad_input_ref[0, :2, :2]}")

    # Forward pass and compute gradients
    output_ref_grad = xielu_ref(grad_input_ref)
    output_xielu_grad = xielu(grad_input_xielu)

    # Create a dummy loss (sum of outputs)
    loss_ref = output_ref_grad.sum()
    loss_xielu = output_xielu_grad.sum()

    # Backward pass
    loss_ref.backward()
    loss_xielu.backward()

    # Get gradients
    grad_ref = grad_input_ref.grad
    grad_xielu = grad_input_xielu.grad

    print(f"\nReference gradients sample (F64):\n{grad_ref[0, :2, :2]}")
    print(f"XIELU gradients sample ({dtype}):\n{grad_xielu[0, :2, :2]}")

    # Compare XIELU vs Reference gradients (convert test to F64)
    grad_diff_xielu = torch.abs(grad_ref - grad_xielu.double())
    max_grad_diff_xielu = torch.max(grad_diff_xielu)
    mean_grad_diff_xielu = torch.mean(grad_diff_xielu)

    print(f"\nXIELU vs Reference gradients:")
    print(f"Max absolute gradient difference: {max_grad_diff_xielu.item():.2e}")
    print(f"Mean absolute gradient difference: {mean_grad_diff_xielu.item():.2e}")

    if max_grad_diff_xielu > tols["threshold"]:
        max_indices = torch.where(grad_diff_xielu == max_grad_diff_xielu)
        print(f"Max gradient diff locations: {list(zip(*[idx.tolist() for idx in max_indices]))[:3]}")
        for i, (b, h, w) in enumerate(zip(*max_indices)):
            if i >= 3:
                break
            print(f"  [{b},{h},{w}]: ref_grad={grad_ref[b,h,w]:.6f}, xielu_grad={grad_xielu[b,h,w]:.6f}, diff={grad_diff_xielu[b,h,w]:.6f}")

    # Run formal gradcheck
    print(f"\n3. Running formal gradcheck ...")
    tiny_input = input_test.clone().detach().requires_grad_(True)

    def xielu_wrapper(x):
        return xielu(x)

    gradcheck(xielu_wrapper, (tiny_input,), eps=tols["grad_eps"], rtol=tols["grad_rtol"], atol=tols["grad_atol"])

    # Tolerance checks for manual gradients (convert test to F64)
    torch.testing.assert_close(grad_xielu.double(), grad_ref, rtol=tols["grad_rtol"], atol=tols["grad_atol"])
    print("All gradient tests passed")

    print("\nAll detailed tests passed!")


@pytest.mark.parametrize("with_vector_loads", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_reduced_precision(xielu_setup, with_vector_loads, dtype):
    """Test XIELU with different precisions against F64 reference"""
    alpha_p_init = xielu_setup["alpha_p_init"]
    alpha_n_init = xielu_setup["alpha_n_init"]
    beta = xielu_setup["beta"]
    eps = xielu_setup["eps"]
    device = xielu_setup["device"]

    # Always use F64 as reference
    ref_dtype = torch.float64
    hiddendim_large = 1024

    tols = get_tolerances(dtype)

    # Create F64 reference input
    input_ref = torch.randn(
        NBATCH,
        NSEQ,
        hiddendim_large,
        device=device,
        dtype=ref_dtype,
        requires_grad=True,
    )

    # Convert to test dtype
    input_test = input_ref.clone().detach().to(dtype).requires_grad_(True)

    # F64 reference implementation
    xielu_ref = XIELUPy(alpha_p_init, alpha_n_init, beta, eps, device, ref_dtype)

    # Test dtype implementation
    xielu_test = XIELU(
        alpha_p_init,
        alpha_n_init,
        beta,
        eps,
        device,
        dtype,
        with_vector_loads=with_vector_loads,
    )

    print(f"\nTesting dtype={dtype} with with_vector_loads={with_vector_loads}")
    print(f"Reference: {ref_dtype}")
    print(f"Tolerances: rtol={tols['rtol']:.2e}, atol={tols['atol']:.2e}")

    # Forward pass
    out_test = xielu_test.forward(input_test)
    out_ref = xielu_ref.forward(input_ref)

    print(f"Test output ({dtype}):\n{out_test[:1, :2, :5]}")
    print(f"Reference output ({ref_dtype}):\n{out_ref[:1, :2, :5]}")

    # Compare outputs (convert test to F64 for comparison)
    mean_output_error = (out_test.double() - out_ref).abs().mean()
    print(f"Mean output error: {mean_output_error.item():.6e}")

    # Backward pass
    grad_output_test = torch.ones_like(out_test)
    out_test.backward(grad_output_test, retain_graph=True)

    grad_output_ref = torch.ones_like(out_ref)
    out_ref.backward(grad_output_ref, retain_graph=True)

    # Check input gradient (convert test to F64 for comparison)
    mean_input_grad_error = (input_test.grad.double() - input_ref.grad).abs().mean()
    print(f"Mean input.grad error: {mean_input_grad_error.item():.6e}")

    # Check alpha_p gradient
    print(f"alpha_p.grad (ref): {xielu_ref.alpha_p.grad}")
    print(f"alpha_p.grad (test): {xielu_test.alpha_p.grad}")
    # Note: alpha gradients are always accumulated in F32 for half/bfloat16, so convert appropriately
    alpha_p_grad_test = xielu_test.alpha_p.grad.double() if xielu_test.alpha_p.grad.dtype != torch.float64 else xielu_test.alpha_p.grad
    mean_alpha_p_error = (xielu_ref.alpha_p.grad - alpha_p_grad_test).abs().mean()
    print(f"Mean alpha_p.grad error: {mean_alpha_p_error.item():.6e}")

    # Check alpha_n gradient
    print(f"alpha_n.grad (ref): {xielu_ref.alpha_n.grad}")
    print(f"alpha_n.grad (test): {xielu_test.alpha_n.grad}")
    alpha_n_grad_test = xielu_test.alpha_n.grad.double() if xielu_test.alpha_n.grad.dtype != torch.float64 else xielu_test.alpha_n.grad
    mean_alpha_n_error = (xielu_ref.alpha_n.grad - alpha_n_grad_test).abs().mean()
    print(f"Mean alpha_n.grad error: {mean_alpha_n_error.item():.6e}")

    # Assert tolerances based on dtype
    assert mean_output_error < tols["atol"], f"Output error too large: {mean_output_error} (threshold: {tols['atol']})"
    assert mean_input_grad_error < tols["grad_atol"], f"Input gradient error too large: {mean_input_grad_error} (threshold: {tols['grad_atol']})"
    assert mean_alpha_p_error < tols["grad_atol_alpha"], f"alpha_p gradient error too large: {mean_alpha_p_error} (threshold: {tols['grad_atol_alpha']})"
    assert mean_alpha_n_error < tols["grad_atol_alpha"], f"alpha_n gradient error too large: {mean_alpha_n_error} (threshold: {tols['grad_atol_alpha']})"

    print(f"Precision test passed for {dtype}!")
