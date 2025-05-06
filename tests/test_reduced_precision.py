import unittest
import torch
from torch.autograd import gradcheck
from xielu.ops.wrappers import XIELUPy, XIELUfn, XIELU

import torch.nn.functional as F

NBATCH = 1
NSEQ = 32
HIDDENDIM = 32

alpha_p_init = 0.8111
alpha_n_init = 0.8111
beta = 0.5
eps = 1e-6

device = torch.device("cuda")
ref_dtype = torch.float32
dtype = torch.bfloat16

input = torch.randn(
    NBATCH,
    NSEQ,
    HIDDENDIM,
    device=device,
    dtype=ref_dtype,
    requires_grad=True,
)

input_fp16 = input.clone().detach().to(torch.bfloat16).requires_grad_(True)

xielu_py = XIELUPy(alpha_p_init, alpha_n_init,
                   beta, eps, device, input.dtype)

xielu = XIELU(alpha_p_init, alpha_n_init,
              beta, eps, device, input_fp16.dtype)

out_cuda = xielu.forward(input_fp16)
out_py = xielu_py.forward(input)

print(out_cuda)
print(out_py)

print(out_cuda - out_py)

grad_output_cuda = torch.ones_like(out_cuda)
out_cuda.backward(grad_output_cuda, retain_graph=True)

grad_output_py = torch.ones_like(out_py)
out_py.backward(grad_output_py, retain_graph=True)

print(input_fp16.grad)
print(input.grad)
print(input_fp16.grad - input.grad)
