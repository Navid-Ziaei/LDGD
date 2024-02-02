import torch
import torch.nn as nn
import torch.optim as optim
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ExactGP


class GPLVM(nn.Module):
    def __init__(self, Y, latent_dim):
        super(GPLVM, self).__init__()

        self.Y = Y
        self.num_data, self.data_dim = Y.shape

        # Initial guess for the latent variables (using PCA)
        U, S, V = torch.svd(Y)
        self.X = nn.Parameter(U[:, :latent_dim] @ torch.diag(S[:latent_dim]))

        # GP model
        self.gp = ExactGPLVM(self.X, self.Y)

    def optimize(self, num_epochs=1000, lr=1e-2):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = self.gp(self.X)
            loss = -mll(output, self.Y.t())
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    def forward(self, X_new):
        return self.gp(X_new)


class ExactGPLVM(ExactGP):
    def __init__(self, X, Y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(ExactGPLVM, self).__init__(X, Y, likelihood)

        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return MultivariateNormal(mean_x, covar_x)