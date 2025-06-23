import torch


def convert_data_to_timelagged(X: torch.Tensor, lag: int) -> torch.Tensor:
    """Convert data from shape: (batch_size, num_variates, timesteps, num_grid_points) 
    to shape (batch_size*n_fragments, num_variates, lag+1, num_grid_points)


    Args:
        X (torch.Tensor): Input tensor
        lag (int): lag to use
    Returns:
        torch.Tensor: Converted tensor
    """
    if len(X.shape) == 3:
        X = X.unsqueeze(2)
    n_samples, num_variates, timesteps, num_grid_points = X.shape

    n_fragments_per_sample = timesteps - lag
    n_fragments = n_samples * n_fragments_per_sample

    n_fragments_per_sample = timesteps - lag
    n_fragments = n_samples * n_fragments_per_sample

    i = torch.arange(timesteps - lag)
    j = torch.arange(lag+1)
    indices = i[:, None] + j

    X_reshaped = X[:, :, indices]
    X_reshaped = X_reshaped.permute((0, 2, 1, 3, 4)).reshape(
        n_fragments, num_variates, lag+1, num_grid_points)

    return X_reshaped



def cdsd_convert_timelagged(X: torch.Tensor, lag: int) -> torch.Tensor:
    """Convert data from shape: (batch_size, num_variates, timesteps, num_grid_points) 
    to shape (batch_size*n_fragments, num_variates, lag+1, num_grid_points)


    Args:
        X (torch.Tensor): Input tensor
        lag (int): lag to use
    Returns:
        torch.Tensor: Converted tensor
    """
    if len(X.shape) == 3:
        X = X.unsqueeze(2)
    n_samples, num_variates, timesteps, num_grid_points = X.shape

    n_fragments_per_sample = timesteps - lag
    n_fragments = n_samples * n_fragments_per_sample

    n_fragments_per_sample = timesteps - lag
    n_fragments = n_samples * n_fragments_per_sample

    i = torch.arange(timesteps - lag)
    j = torch.arange(lag+1)
    indices = i[:, None] + j

    X_reshaped = X[:, :, indices]
    X_reshaped = X_reshaped.permute((0, 2, 1, 3, 4)).reshape(
        n_fragments, num_variates, lag+1, num_grid_points)
    X_curr = X_reshaped[:,:,-1,:]

    return X_reshaped, X_curr