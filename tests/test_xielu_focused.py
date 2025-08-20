#!/usr/bin/env python3

import torch
from xielu.ops.wrappers import XIELUPy, XIELU

NBATCH = 1
NSEQ = 32
HIDDENDIM = 32


def test_xielu_detailed():
    """Test XIELU class with detailed element-wise comparison"""

    # Setup
    alpha_p_init = 0.8111
    alpha_n_init = 0.8111
    beta = 0.5
    eps = 1e-6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a smaller test tensor for easier debugging
    test_input = torch.randn(
        NBATCH, NSEQ, HIDDENDIM, device=device, dtype=torch.float64, requires_grad=True)

    # Create reference (Python) and test (XIELU) implementations
    xielu_py = XIELUPy(alpha_p_init, alpha_n_init, beta,
                       eps, device, test_input.dtype)
    xielu = XIELU(alpha_p_init, alpha_n_init, beta,
                  eps, device, test_input.dtype)
    xielu_compiled = torch.compile(
        XIELU(alpha_p_init, alpha_n_init, beta, eps, device, test_input.dtype))

    print("Testing XIELU implementations...")
    print(f"Input shape: {test_input.shape}")
    print(f"Device: {device}")
    print(f"Input sample values:\n{test_input}")
    print()

    try:
        # Test regular XIELU
        print("1. Testing regular XIELU...")
        print(test_input.shape)
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
            # Show first 5
            print(
                f"Max diff locations: {list(zip(*[idx.tolist() for idx in max_indices]))[:5]}")
            for i, (b, h, w) in enumerate(zip(*max_indices)):
                if i >= 3:
                    break  # Only show first 3
                print(
                    f"  [{b},{h},{w}]: py={output_py[b,h,w]:.6f}, xielu={output_xielu[b,h,w]:.6f}, diff={diff[b,h,w]:.6f}")

        torch.testing.assert_close(
            output_py, output_xielu, rtol=1e-6, atol=1e-6)
        print("Regular XIELU test passed")

    except Exception as e:
        print(f"Regular XIELU test failed: {e}")
        return False

    try:
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
            print(
                f"Max diff locations: {list(zip(*[idx.tolist() for idx in max_indices]))[:5]}")
            for i, (b, h, w) in enumerate(zip(*max_indices)):
                if i >= 3:
                    break
                print(
                    f"  [{b},{h},{w}]: py={output_py[b,h,w]:.6f}, compiled={output_compiled[b,h,w]:.6f}, diff={diff[b,h,w]:.6f}")

        torch.testing.assert_close(
            output_py, output_compiled, rtol=1e-6, atol=1e-6)
        print("Compiled XIELU test passed")

    except Exception as e:
        print(f"Compiled XIELU test failed: {e}")
        return False

    try:
        # Test gradient computation with detailed comparison
        print("\n3. Testing gradients...")

        # Create fresh inputs for gradient testing (smaller for easier inspection)
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
        print(
            f"Max absolute gradient difference: {max_grad_diff_xielu.item():.2e}")
        print(
            f"Mean absolute gradient difference: {mean_grad_diff_xielu.item():.2e}")

        if max_grad_diff_xielu > 1e-6:
            max_indices = torch.where(grad_diff_xielu == max_grad_diff_xielu)
            print(
                f"Max gradient diff locations: {list(zip(*[idx.tolist() for idx in max_indices]))[:3]}")
            for i, (b, h, w) in enumerate(zip(*max_indices)):
                if i >= 3:
                    break
                print(
                    f"  [{b},{h},{w}]: py_grad={grad_py[b,h,w]:.6f}, xielu_grad={grad_xielu[b,h,w]:.6f}, diff={grad_diff_xielu[b,h,w]:.6f}")

        # Compare Compiled vs Python gradients
        grad_diff_compiled = torch.abs(grad_py - grad_compiled)
        max_grad_diff_compiled = torch.max(grad_diff_compiled)
        mean_grad_diff_compiled = torch.mean(grad_diff_compiled)

        print(f"\nCompiled vs Python gradients:")
        print(
            f"Max absolute gradient difference: {max_grad_diff_compiled.item():.2e}")
        print(
            f"Mean absolute gradient difference: {mean_grad_diff_compiled.item():.2e}")

        if max_grad_diff_compiled > 1e-6:
            max_indices = torch.where(
                grad_diff_compiled == max_grad_diff_compiled)
            print(
                f"Max gradient diff locations: {list(zip(*[idx.tolist() for idx in max_indices]))[:3]}")
            for i, (b, h, w) in enumerate(zip(*max_indices)):
                if i >= 3:
                    break
                print(
                    f"  [{b},{h},{w}]: py_grad={grad_py[b,h,w]:.6f}, compiled_grad={grad_compiled[b,h,w]:.6f}, diff={grad_diff_compiled[b,h,w]:.6f}")

        # Optional: Run formal gradcheck
        print(f"\n4. Running formal gradcheck ...")
        tiny_input = test_input.clone().detach().requires_grad_(True)
        from torch.autograd import gradcheck

        def xielu_wrapper(x):
            return xielu(x)

        def xielu_compiled_wrapper(x):
            return xielu_compiled(x)

        gradcheck(xielu_wrapper, (tiny_input,), eps=1e-3, rtol=1e-4, atol=1e-4)
        gradcheck(xielu_compiled_wrapper, (tiny_input,),
                  eps=1e-3, rtol=1e-4, atol=1e-4)

        # Tolerance checks for manual gradients
        torch.testing.assert_close(grad_py, grad_xielu, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(
            grad_py, grad_compiled, rtol=1e-4, atol=1e-4)
        print("All gradient tests passed")

    except Exception as e:
        print(f"Gradient test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\nAll tests passed!")
    return True


if __name__ == "__main__":
    success = test_xielu_detailed()
    exit(0 if success else 1)
