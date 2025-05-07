import unittest
import torch
from torch.autograd import gradcheck
from xielu.ops.wrappers import XIELUPy, XIELUfn, XIELU

import torch.nn.functional as F

NBATCH = 1
NSEQ = 32
HIDDENDIM = 1024

alpha_p_init = 0.8111
alpha_n_init = 0.8111
beta = 0.5
eps = 1e-6

with_vector_loads = True

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

input_reduced = input.clone().detach().to(dtype).requires_grad_(True)

xielu_py = XIELUPy(alpha_p_init, alpha_n_init,
                   beta, eps, device, input.dtype)

xielu = XIELU(alpha_p_init, alpha_n_init,
              beta, eps, device, input_reduced.dtype, with_vector_loads=with_vector_loads)

out_cuda = xielu.forward(input_reduced)
out_py = xielu_py.forward(input)

print(out_cuda)
print(out_py)
print("mean output error...")
print((out_cuda - out_py).abs().mean())

grad_output_cuda = torch.ones_like(out_cuda)
out_cuda.backward(grad_output_cuda, retain_graph=True)

grad_output_py = torch.ones_like(out_py)
out_py.backward(grad_output_py, retain_graph=True)

print("mean input.grad error...")
print((input_reduced.grad - input.grad).abs().mean())

print("alpha_p error...")
print(xielu_py.alpha_p.grad)
print(xielu.alpha_p.grad)
print((xielu_py.alpha_p.grad - xielu.alpha_p.grad).abs().mean())

print("alpha_n error...")
print(xielu_py.alpha_n.grad)
print(xielu.alpha_n.grad)
print((xielu_py.alpha_n.grad - xielu.alpha_n.grad).abs().mean())
