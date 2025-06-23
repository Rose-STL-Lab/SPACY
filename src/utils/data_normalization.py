import torch


def standardize(X, dim, eps=1e-5):
    """
    Standardize normalizes input data by subtracting the mean and dividing by the standard
    deviation along the specified dimension.

    :param X: A tensor containing the input data that you want to standardize. 
    :param dim: An int that determines which dimension to normalize along
    :param eps: The `eps` parameter in the `standardize` function is a small value added to the
    denominator to prevent division by zero when calculating the standard deviation. Default: 1e-5
    :return: normalized data
    """
    with torch.no_grad():
        return (X-torch.mean(X, dim=dim).unsqueeze(dim))/torch.sqrt(torch.var(X, dim=dim).unsqueeze(dim) + eps)
