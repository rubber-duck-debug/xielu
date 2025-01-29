import torch
import torch.nn.functional as F


class XIELUPy(torch.nn.Module):
    def __init__(self, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super(XIELUPy, self).__init__()
        self.beta = beta
        self.alpha_p = torch.nn.Parameter(
            torch.log(torch.exp(torch.tensor(alpha_p_init)) - 1).unsqueeze(0))
        self.alpha_n = torch.nn.Parameter(
            torch.log(torch.exp(torch.tensor(alpha_n_init - self.beta)) - 1).unsqueeze(0))
        self.eps = torch.tensor(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        return torch.where(x > 0,
                           alpha_p * x * x + self.beta * x,
                           alpha_n * torch.expm1(torch.min(x, self.eps)) - alpha_n * x + self.beta * x)

    def forward_test(self, x: torch.Tensor, a_p, a_n) -> torch.Tensor:
        alpha_p = F.softplus(a_p)
        alpha_n = self.beta + F.softplus(a_n)
        return torch.where(x > 0,
                           alpha_p * x * x + self.beta * x,
                           alpha_n * torch.expm1(torch.min(x, self.eps)) - alpha_n * x + self.beta * x)


class XIELU(torch.nn.Module):
    def __init__(self, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=1e-6):
        super(XIELU, self).__init__()

        self.cuda_obj = torch.classes.xielu.XIELU()
        self.eps = eps
        self.beta = beta
        self.alpha_p = torch.nn.Parameter(
            torch.log(torch.exp(torch.tensor(alpha_p_init)) - 1).unsqueeze(0))
        self.alpha_n = torch.nn.Parameter(
            torch.log(torch.exp(torch.tensor(alpha_n_init - self.beta)) - 1).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cuda_obj.forward(x, self.alpha_p, self.alpha_n, self.beta, self.eps)

    def forward_test(self, x: torch.Tensor, a_p, a_n) -> torch.Tensor:
        return self.cuda_obj.forward(x, a_p, a_n, self.beta, self.eps)
