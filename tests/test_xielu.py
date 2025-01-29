import unittest
import torch
from torch.autograd import gradcheck
from xielu.ops.wrappers import XIELUPy, XIELU

import torch.nn.functional as F

NBATCH = 4
NSEQ = 16
HIDDENDIM = 2

class TestXIELU(unittest.TestCase):
    def setUp(self):
        self.alpha_p_init = 0.8
        self.alpha_n_init = 0.8
        self.beta = 0.5
        self.eps = 1e-6

        self.dtypes = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
        self.device = torch.device("cuda")

        self.input = torch.randn(
            NBATCH,
            NSEQ,
            HIDDENDIM,
            device=self.device,
            dtype=torch.float64,
            requires_grad=True,
        )

        self.xielu_py = (
            XIELUPy(self.alpha_p_init, self.alpha_n_init, self.beta, self.eps)
            .to(self.device)
            .to(self.input.dtype)
        )
        self.xielu = (
            XIELU(self.alpha_p_init, self.alpha_n_init, self.beta, self.eps)
            .to(self.device)
            .to(self.input.dtype)
        )

    def run_forward_comparison(self, input_tensor):
        output_py = self.xielu_py(input_tensor)
        output_cuda = self.xielu(input_tensor)
        torch.testing.assert_close(output_py, output_cuda, rtol=1e-6, atol=1e-6)

    def test_forward(self):
        self.run_forward_comparison(self.input)

    def run_gradcheck(self, model, inputs, dtype=torch.float64, eps=1e-5, rtol=1e-5, atol=1e-5):
        inputs = tuple(inp.clone().detach().to(dtype).requires_grad_(True) for inp in inputs)
        gradcheck(model, inputs, eps=eps, rtol=rtol, atol=atol, fast_mode=True)

    def test_gradients(self):
        self.run_gradcheck(self.xielu_py, (self.input,))
        self.run_gradcheck(self.xielu, (self.input,))

    def test_alpha_p_alpha_n_gradients(self):
        self.run_gradcheck(self.xielu_py.forward_test, (self.input, self.xielu_py.alpha_p, self.xielu_py.alpha_n))
        self.run_gradcheck(self.xielu.forward_test, (self.input, self.xielu.alpha_p, self.xielu.alpha_n))

    def test_edge_cases(self):
        edge_inputs = {
            "zero": torch.zeros_like(self.input, dtype=torch.float64, requires_grad=True),
            "large": torch.full_like(self.input, 1e10, dtype=torch.float64, requires_grad=True),
            "small": torch.full_like(self.input, 1e-10, dtype=torch.float64, requires_grad=True),
            "negative": torch.full_like(self.input, -1.0, dtype=torch.float64, requires_grad=True),
        }
        for case, input_tensor in edge_inputs.items():
            with self.subTest(case=case):
                self.run_forward_comparison(input_tensor)

    #def test_alpha_p_alpha_n_dtypes(self):
    #    """Test alpha_p and alpha_n gradients with different dtypes."""
    #    for dtype in self.dtypes:
    #        with self.subTest(dtype=dtype):
    #            if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
    #                self.skipTest("bfloat16 is not supported on this GPU.")
    #            self.run_gradcheck(self.xielu.forward_test, (self.input, self.xielu.alpha_p, self.xielu.alpha_n), dtype,
    #                atol=1e-3 if dtype in [torch.float16, torch.bfloat16] else 1e-5)

if __name__ == "__main__":
    unittest.main()
