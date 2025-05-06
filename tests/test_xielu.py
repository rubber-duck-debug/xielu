import unittest
import torch
from torch.autograd import gradcheck
from xielu.ops.wrappers import XIELUPy, XIELUfn, XIELU

import torch.nn.functional as F

NBATCH = 1
NSEQ = 32
HIDDENDIM = 32


class TestXIELU(unittest.TestCase):
    def setUp(self):
        self.alpha_p_init = 0.8111
        self.alpha_n_init = 0.8111
        self.beta = 0.5
        self.eps = 1e-6

        self.device = torch.device("cuda")

        self.input = torch.randn(
            NBATCH,
            NSEQ,
            HIDDENDIM,
            device=self.device,
            dtype=torch.float64,
            requires_grad=True,
        )

        self.xielu_py = XIELUPy(self.alpha_p_init, self.alpha_n_init,
                                self.beta, self.eps, self.device, self.input.dtype)
        self.xielu_fn = XIELUfn(self.alpha_p_init, self.alpha_n_init,
                                self.beta, self.eps, self.device, self.input.dtype)
        self.xielu = XIELU(self.alpha_p_init, self.alpha_n_init,
                           self.beta, self.eps, self.device, self.input.dtype)

    def run_forward_comparison(self, input_tensor):
        output_py = self.xielu_py(input_tensor)
        output_fn = self.xielu_fn(input_tensor)
        output_cuda = self.xielu(input_tensor)
        torch.testing.assert_close(output_py, output_fn, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            output_py, output_cuda, rtol=1e-6, atol=1e-6)

    def test_forward(self):
        self.run_forward_comparison(self.input)

    def run_gradcheck(self, model, inputs, dtype=torch.float64, eps=1e-3, rtol=1e-4, atol=1e-4):
        inputs = tuple(inp.clone().detach().to(
            dtype).requires_grad_(True) for inp in inputs)
        gradcheck(model, inputs, eps=1e-3, rtol=rtol,
                  atol=atol, fast_mode=False)

    def test_gradients(self):
        self.run_gradcheck(self.xielu, (self.input,))
        self.run_gradcheck(self.xielu_py, (self.input,))
        self.run_gradcheck(self.xielu_fn, (self.input,))

    def test_alpha_p_alpha_n_gradients(self):
        self.run_gradcheck(self.xielu_py.forward_inference, (self.input,
                           self.xielu_py.alpha_p, self.xielu_py.alpha_n))
        self.run_gradcheck(self.xielu_fn.forward_inference,
                           (self.input, self.xielu.alpha_p, self.xielu.alpha_n))
        self.run_gradcheck(self.xielu.forward_inference,
                           (self.input, self.xielu.alpha_p, self.xielu.alpha_n))

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


if __name__ == "__main__":
    unittest.main()
