from gpytorch.models.gplvm.latent_variable import *
from gpytorch.models.gplvm.bayesian_gplvm import BayesianGPLVM
from gpytorch.models.approximate_gp import ApproximateGP
from matplotlib import pyplot as plt
from tqdm.notebook import trange
from gpytorch.means import ZeroMean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import NormalPrior
from gpytorch.variational import VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel
import numpy as np
import gpytorch
import torch
from gpytorch.variational.cholesky_variational_distribution import CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gp_project_pytorch.strategy import VariationalStrategyJointGPLVM

def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q=latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:, :latent_dim]))


class GPLVM(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPLVM, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class bGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False, Y=None):
        self.n = n
        self.batch_shape = torch.Size([data_dim])

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs, q_u, learn_inducing_locations=True)

        # Define prior for X
        X_prior_mean = torch.zeros(n, latent_dim)  # shape: N x Q
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))

        # Initialise X with PCA or randn
        if pca == True:
            X_init = _init_pca(Y, latent_dim)  # Initialise X to PCA
        else:
            X_init = torch.nn.Parameter(torch.randn(n, latent_dim))

        # LatentVariable (c)
        X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=latent_dim))

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)


class JointGPLVM(ApproximateGP):
    def __init__(self, latent_dim, num_data_points, num_classes, continuous_output_dim, n_inducing):
        # Using Variational strategy here as an example
        self.n = num_data_points
        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs_cont = torch.randn(continuous_output_dim, n_inducing, latent_dim)
        self.inducing_inputs_cls = torch.randn(num_classes, n_inducing, latent_dim)

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u = CholeskyVariationalDistribution(n_inducing)

        variational_strategy = VariationalStrategyJointGPLVM(self,
                                                             inducing_points_count = self.inducing_inputs_cont,
                                                             inducing_points_cat = self.inducing_inputs_cls,
                                                             variational_distribution = q_u,
                                                             learn_inducing_locations=True)

        # Define prior for X
        X_prior_mean = torch.zeros(self.n, latent_dim)  # shape: N x Q
        prior_x = NormalPrior(X_prior_mean, torch.ones_like(X_prior_mean))

        # Initialise X with PCA or randn
        X_init = torch.nn.Parameter(torch.randn(self.n, latent_dim))

        # LatentVariable (c)
        X = VariationalLatentVariable(num_data_points, continuous_output_dim, latent_dim, X_init, prior_x)

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(variational_strategy)

        # inducing_points = torch.randn(latent_dim, num_data_points)
        # variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_data_points)
        # variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, variational_distribution)

        # super(JointGPLVM, self).__init__(variational_strategy)

        # Assigning Latent Variable
        self.X = X

        self.mean_module_continuous = gpytorch.means.ConstantMean()
        self.covar_module_continuous = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.mean_module_classification = gpytorch.means.ConstantMean()
        self.covar_module_classification = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        # For continuous measurements
        mean_x_continuous = self.mean_module_continuous(x)
        covar_x_continuous = self.covar_module_continuous(x)
        mvn_continuous = gpytorch.distributions.MultivariateNormal(mean_x_continuous, covar_x_continuous)

        # For classification
        mean_x_classification = self.mean_module_classification(x)
        covar_x_classification = self.covar_module_classification(x)
        mvn_classification = gpytorch.distributions.MultivariateNormal(mean_x_classification, covar_x_classification)

        return mvn_continuous, mvn_classification

    def sample_latent_variable(self):
        sample = self.X()
        return sample


