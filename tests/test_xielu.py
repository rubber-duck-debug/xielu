import unittest
import torch
from torch.autograd import gradcheck
from xielu.ops.wrappers import XIELUPy, XIELU

import torch.nn.functional as F

class TestXIELU(unittest.TestCase):
    def setUp(self):
        self.alpha_p_init = 0.8
        self.alpha_n_init = 0.8
        self.beta = 0.5
        self.eps = 1e-6

        self.device = torch.device("cuda")

        self.xielu_py = XIELUPy(self.alpha_p_init, self.alpha_n_init, self.beta, self.eps).to(self.device)
        self.xielu = XIELU(self.alpha_p_init, self.alpha_n_init, self.beta, self.eps).to(self.device)
        self.input = torch.randn(5, 7, device=self.device, requires_grad=True)

    def test_forward(self):
        output_py = self.xielu_py(self.input)
        output_cuda = self.xielu(self.input)
        torch.testing.assert_close(output_py, output_cuda, atol=1e-6)

    def test_gradients(self):
        input_py = self.input.clone().detach().requires_grad_(True)
        input_cuda = self.input.clone().detach().requires_grad_(True)

        output_py = self.xielu_py(input_py)
        output_cuda = self.xielu(input_cuda)

        gradcheck(self.xielu_py, (input_py,), eps=1e-6, atol=1e-4)
        gradcheck(self.xielu, (input_cuda,), eps=1e-6, atol=1e-4)

if __name__ == '__main__':
    unittest.main()
