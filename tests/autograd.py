import torch
from xielu.ops.wrappers import XIELU, XIELUPy


def finite_difference_gradient(func, h=1e-5, **kwargs):
    """
    Compute finite difference gradients for all tensors requiring gradients within the arguments of a function.

    Parameters:
        func: Callable
            The function to compute the gradient of. It should accept `**kwargs`.
        **kwargs: dict
            Keyword arguments to pass to `func`, including tensors requiring gradients.
        h: float
            The step size for finite differences.

    Returns:
        gradients: tuple
            A tuple containing gradients for all tensors in `args` and `kwargs` that require gradients.
    """

    # Helper function to identify tensors requiring gradients
    def get_tensors_with_grads(**kwargs):
        tensors = []
        names = []
        kwargs_keys = []

        for name, tensor in kwargs.items():
            if tensor.requires_grad:
                tensors.append(tensor)
                names.append(name)

        return names, tensors

    # Get tensors that require gradients and their locations
    names, tensors = get_tensors_with_grads(**kwargs)

    # Initialize a dictionary to store gradients for each tensor
    gradients = {}

    # Compute finite difference gradients for each tensor
    for name, tensor in zip(names, tensors):
        # Initialize gradient tensor
        gradient = torch.zeros_like(tensor)

        # Compute gradients using finite difference method
        for i in range(tensor.numel()):
            perturb = torch.zeros_like(tensor)
            perturb.view(-1)[i] = h

            # Create new arguments with perturbed tensors
            new_kwargs = kwargs.copy()

            new_kwargs[name] = tensor + perturb

            f_x_plus_h = func(**new_kwargs).sum()

            new_kwargs[name] = tensor - perturb

            f_x_minus_h = func(**new_kwargs).sum()

            # Compute finite difference approximation
            gradient.view(-1)[i] = (f_x_plus_h - f_x_minus_h) / (2 * h)

        # Store gradient in the dictionary
        gradients[name] = gradient

    return gradients


def rel_error(tensor1, tensor2, eps=1e-7):
    """
    Compute the relative error between two tensors

    Parameters:
        tensor1: torch.Tensor
        tensor2: torch.Tensor
        eps: desired precision

    Returns:
        relative_error: torch.Tensor
            relative error between two tensors.
    """
    relative_error = torch.abs(tensor1 - tensor2) / torch.maximum(
        torch.tensor(eps), torch.abs(tensor1) + torch.abs(tensor2)
    )
    return relative_error

# TODO
