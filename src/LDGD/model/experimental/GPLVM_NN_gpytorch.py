from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from matplotlib import pyplot as plt
from tqdm.notebook import trange
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.variational import VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel
import numpy as np
from torch import nn
import torch
import gpytorch

class VariationalNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VariationalNN, self).__init__()
        # Simple architecture for demonstration purposes
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, output_dim)
        self.fc2_sigma = nn.Linear(hidden_dim, output_dim)
        self.activation_softplus = nn.Softplus()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        mu = self.fc2_mu(x)
        sigma = self.activation_softplus(self.fc2_sigma(x))  # Ensure positivity
        return mu, sigma


class NeuralNetworkVariationalDistribution(CholeskyVariationalDistribution):
    def __init__(self, variational_nn):
        super(NeuralNetworkVariationalDistribution, self).__init__()
        self.variational_nn = variational_nn

    def forward(self, x):
        mu, sigma = self.variational_nn(x)
        return MultivariateNormal(mu, torch.diag_embed(sigma))
