import torch
import torch.nn.functional as F


@torch.no_grad()
def create_xielu_params(alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6, device=None, dtype=None):
    dev_kwargs = {"device": device, "dtype": dtype}
    alpha_p = torch.nn.Parameter(
        torch.log(torch.exp(torch.tensor(alpha_p_init, **dev_kwargs)) - 1).unsqueeze(0))
    alpha_n = torch.nn.Parameter(
        torch.log(torch.exp(torch.tensor(alpha_n_init - beta, **dev_kwargs)) - 1).unsqueeze(0))
    beta = torch.tensor(beta, **dev_kwargs)
    eps = torch.tensor(eps, **dev_kwargs)
    return alpha_p, alpha_n, beta, eps


class XIELUPy(torch.nn.Module):
    def __init__(self, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6, device=None, dtype=None):
        super().__init__()
        self.alpha_p, self.alpha_n, self.beta, self.eps = create_xielu_params(
            alpha_p_init, alpha_n_init, beta, eps, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        return torch.where(x > 0,
                           alpha_p * x * x + self.beta * x,
                           alpha_n * torch.expm1(torch.min(x, self.eps)) - alpha_n * x + self.beta * x)

    def forward_inference(self, x: torch.Tensor, a_p, a_n) -> torch.Tensor:
        alpha_p = F.softplus(a_p)
        alpha_n = self.beta + F.softplus(a_n)
        return torch.where(x > 0,
                           alpha_p * x * x + self.beta * x,
                           alpha_n * torch.expm1(torch.min(x, self.eps)) - alpha_n * x + self.beta * x)


class XIELUfn(torch.nn.Module):
    class XIELUfunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, alpha_p, alpha_n, beta, eps):
            ctx.save_for_backward(x, alpha_p, alpha_n, beta, eps)
            alpha_p = F.softplus(alpha_p)
            alpha_n = beta + F.softplus(alpha_n)
            output = torch.where(x > 0,
                                 alpha_p * x * x + beta * x,
                                 alpha_n * torch.expm1(torch.clamp_max(x, eps)) - alpha_n * x + beta * x)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            x, alpha_p, alpha_n, beta, eps = ctx.saved_tensors
            p = F.softplus(alpha_p)
            n = beta + F.softplus(alpha_n)

            grad_input = torch.zeros_like(x)
            grad_alpha_p = torch.zeros_like(alpha_p)
            grad_alpha_n = torch.zeros_like(alpha_n)

            positive_mask = x > 0
            negative_mask = x <= 0

            grad_input[positive_mask] = grad_output[positive_mask] * \
                (2 * p * x[positive_mask] + beta)
            grad_input[negative_mask] = grad_output[negative_mask] * \
                (n *
                 torch.exp(torch.clamp_max(x[negative_mask], eps)) - n + beta)

            grad_alpha_p = torch.sum(
                grad_output[positive_mask] * x[positive_mask] * x[positive_mask] * torch.sigmoid(alpha_p))
            grad_alpha_n = torch.sum(grad_output[negative_mask] * (torch.expm1(
                torch.clamp_max(x[negative_mask], eps)) - x[negative_mask]) * torch.sigmoid(alpha_n))

            return grad_input, grad_alpha_p.unsqueeze(0), grad_alpha_n.unsqueeze(0), None, None

    def __init__(self, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6, device=None, dtype=None):
        super().__init__()
        self.alpha_p, self.alpha_n, self.beta, self.eps = create_xielu_params(
            alpha_p_init, alpha_n_init, beta, eps, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.XIELUfunction.apply(x, self.alpha_p, self.alpha_n, self.beta, self.eps)

    def forward_inference(self, x: torch.Tensor, a_p, a_n) -> torch.Tensor:
        return self.XIELUfunction.apply(x, a_p, a_n, self.beta, self.eps)


class XIELU(torch.nn.Module):
    def __init__(self, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6, device=None, dtype=None, with_vector_loads=True):
        super().__init__()
        self.alpha_p, self.alpha_n, self.beta, self.eps = create_xielu_params(
            alpha_p_init, alpha_n_init, beta, eps, device, dtype)
        self.with_vector_loads = with_vector_loads
        self.cuda_obj = torch.classes.xielu.XIELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cuda_obj.forward(x, self.alpha_p, self.alpha_n, self.beta, self.eps, self.with_vector_loads)

    def forward_inference(self, x: torch.Tensor, a_p, a_n) -> torch.Tensor:
        return self.cuda_obj.forward(x, a_p, a_n, self.beta, self.eps, self.with_vector_loads)
