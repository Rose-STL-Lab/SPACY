import torch
import lightning.pytorch as pl
import torch.nn.functional as F
import torch.distributions as D

import numpy as np

class TensorPCA(pl.LightningModule):
    def __init__(self, 
                lag,
                num_nodes,
                nx,
                ny,
                num_variates):
        """
        Initialize the TensorPCA class.

        Parameters:
        - num_nodes (int): Number of reduced nodes (dimensionality along the reduced dimension).
        """
        self.num_nodes = num_nodes
        self.nx = nx
        self.ny = ny
        self.lag = lag
        self.num_variates = num_variates
        self.num_nodes = num_nodes
        self.components_grid_ = None
        self.components_variate_ = None

    def fit(self, X):
        """
        Fit the TensorPCA model to the input tensor X.

        Parameters:
        - X (torch.Tensor): Input tensor of shape (variate, timesteps, grid_size).
        
        Returns:
        - self: Fitted TensorPCA object.
        """
        # Validate input shape
        if X.ndim != 3:
            raise ValueError("Input tensor must have exactly 3 dimensions: (variate, timesteps, grid_size)")

        variate, timesteps, grid_size = X.shape

        # Unfold the tensor along the grid_size dimension
        X_unfolded_grid = X.reshape(variate * timesteps, grid_size)

        # Compute the covariance matrix for the grid dimension
        covariance_matrix_grid = torch.matmul(X_unfolded_grid.T, X_unfolded_grid)

        # Perform eigendecomposition for the grid dimension
        eigenvalues_grid, eigenvectors_grid = torch.linalg.eigh(covariance_matrix_grid)

        # Select the top num_nodes principal components for the grid dimension
        idx_grid = torch.argsort(eigenvalues_grid, descending=True)[:self.num_nodes]
        self.components_grid_ = eigenvectors_grid[:, idx_grid]

        # Project onto the top principal components for the grid dimension
        X_reduced_grid = torch.matmul(X_unfolded_grid, self.components_grid_)

        # Reshape to (variate, timesteps, num_nodes) for variate reduction
        X_reduced_grid = X_reduced_grid.reshape(variate, timesteps, self.num_nodes)

        # Unfold the tensor along the variate dimension
        X_unfolded_variate = X_reduced_grid.permute(1, 2, 0).reshape(timesteps * self.num_nodes, variate)

        # Compute the covariance matrix for the variate dimension
        covariance_matrix_variate = torch.matmul(X_unfolded_variate.T, X_unfolded_variate)

        # Perform eigendecomposition for the variate dimension
        eigenvalues_variate, eigenvectors_variate = torch.linalg.eigh(covariance_matrix_variate)

        # Select the top 1 principal component for the variate dimension
        idx_variate = torch.argsort(eigenvalues_variate, descending=True)[:1]
        self.components_variate_ = eigenvectors_variate[:, idx_variate]

        return self

    def transform(self, X):
        """
        Transform the input tensor X using the fitted TensorPCA model.

        Parameters:
        - X (torch.Tensor): Input tensor of shape (variate, timesteps, grid_size).
        
        Returns:
        - Z (torch.Tensor): Reduced tensor of shape (1, timesteps, num_nodes).
        """
        if self.components_grid_ is None or self.components_variate_ is None:
            raise ValueError("The model has not been fitted yet. Call 'fit' before 'transform'.")

        variate, timesteps, grid_size = X.shape

        # Unfold the tensor along the grid_size dimension
        X_unfolded_grid = X.reshape(variate * timesteps, grid_size)

        # Project onto the top principal components for the grid dimension
        X_reduced_grid = torch.matmul(X_unfolded_grid, self.components_grid_)

        # Reshape to (variate, timesteps, num_nodes) for variate reduction
        X_reduced_grid = X_reduced_grid.reshape(variate, timesteps, self.num_nodes)

        # Unfold the tensor along the variate dimension
        X_unfolded_variate = X_reduced_grid.permute(1, 2, 0).reshape(timesteps * self.num_nodes, variate)

        # Project onto the top principal component for the variate dimension
        Z_unfolded = torch.matmul(X_unfolded_variate, self.components_variate_)
        
        # Reshape back to the reduced tensor form
        Z = Z_unfolded.reshape(1, timesteps, self.num_nodes)
        return Z

    def fit_transform(self, X):
        """
        Fit the TensorPCA model and transform the input tensor X.

        Parameters:
        - X (torch.Tensor): Input tensor of shape (variate, timesteps, grid_size).
        
        Returns:
        - Z (torch.Tensor): Reduced tensor of shape (1, timesteps, num_nodes).
        """
        self.fit(X)
        return self.transform(X)

# Example usage
if __name__ == "__main__":
    # Generate a random tensor with shape (variate, timesteps, grid_size)
    variate, timesteps, grid_size = 2, 10, 100
    num_nodes = 5

    X = torch.rand(variate, timesteps, grid_size)

    # Perform TensorPCA
    tensor_pca = TensorPCA(num_nodes=num_nodes)
    Z = tensor_pca.fit_transform(X)

    print("Original shape:", X.shape)
    print("Reduced shape:", Z.shape)


#     # Perform TensorPCA
#     tensor_pca = TensorPCA(num_nodes=num_nodes)
#     Z = tensor_pca.fit_transform(X)

#     print("Original shape:", X.shape)
#     print("Reduced shape:", Z.shape)
