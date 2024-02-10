import torch
import torch.nn as nn
from linear_operator.operators import CholLinearOperator, TriangularLinearOperator
from torch import Tensor

from gpytorch.variational.variational_strategy import VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import pop_from_cache_ignore_args

from linear_operator import to_dense
from linear_operator.operators import (
    CholLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
)
import gpytorch


class VariationalLatentVariable(nn.Module):
    def __init__(self, n, data_dim, latent_dim, X_init, prior_x, device=None):
        super(VariationalLatentVariable, self).__init__()
        self.data_dim = data_dim
        self.prior_x = prior_x
        self.n = n
        self.data_dim = data_dim
        # G: there might be some issues here if someone calls .cuda() on their BayesianGPLVM
        # after initializing on the CPU

        # Local variational params per latent point with dimensionality latent_dim
        self.q_mu = torch.nn.Parameter(X_init)
        self.q_log_sigma = torch.nn.Parameter(torch.randn(n, latent_dim))
        self.kl_loss = None
        self.kl_loss_list = []

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.to(device=self.device)

    @property
    def q_sigma(self):
        return torch.nn.functional.softplus(self.q_log_sigma)

    def forward(self, num_samples=5):
        # Variational distribution over the latent variable q(x)
        q_x = torch.distributions.Normal(self.q_mu, self.q_sigma)

        kl_per_latent_dim = torch.distributions.kl_divergence(q_x, self.prior_x).sum(axis=0)
        # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum() / self.n  # scalar
        self.kl_loss = kl_per_point / self.data_dim
        self.kl_loss_list.append(self.kl_loss.cpu().detach().numpy())
        return q_x.rsample()

    def kl(self):
        n, q = self.q_mu.shape
        q_x = torch.distributions.Normal(self.q_mu, self.q_sigma)

        # vector of size latent_dim
        kl_per_latent_dim = torch.distributions.kl_divergence(q_x.to, self.prior_x).sum(axis=0)

        kl_per_point = kl_per_latent_dim.sum() / n
        kl_term = kl_per_point / q
        return kl_term


class VariationalLatentVariableNN(nn.Module):
    def __init__(self, n, data_dim, latent_dim, X_init, prior_x, device=None):
        super(VariationalLatentVariableNN, self).__init__()

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.data_dim = data_dim
        self.prior_x = prior_x
        self.n = n
        self.data_dim = data_dim

        hidden_dim = 50

        self.encoder = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 10),
            nn.LeakyReLU(0.2)
        ).to(self.device)

        # latent mean and variance
        self.mean_layer = nn.Linear(10, latent_dim)
        self.logvar_layer = nn.Linear(10, latent_dim)

        # Local variational params per latent point with dimensionality latent_dim
        self.kl_loss = None
        self.kl_loss_list = []

        self.to(device=self.device)

    @property
    def q_sigma(self, yn):
        x = self.encoder(yn)
        log_var = self.logvar_layer(x)
        return torch.nn.functional.softplus(log_var)

    @property
    def q_mu(self, yn):
        x = self.encoder(yn)
        mean = self.mean_layer(x)
        return mean

    def encode(self, y_n):
        x = self.encoder(y_n)
        mean, log_var = self.mean_layer(x), self.logvar_layer(x)
        return mean, log_var

    def forward(self, y_n, num_samples=5):
        mean, log_var = self.encode(y_n)

        # Variational distribution over the latent variable q(x)
        q_x = torch.distributions.Normal(mean, torch.nn.functional.softplus(log_var))

        kl_per_latent_dim = torch.distributions.kl_divergence(q_x, self.prior_x).sum(axis=0)
        # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum() / self.n  # scalar
        self.kl_loss = kl_per_point / self.data_dim
        self.kl_loss_list.append(self.kl_loss.cpu().detach().numpy())
        return q_x.rsample()

    def kl(self):
        n, q = self.q_mu.shape
        q_x = torch.distributions.Normal(self.q_mu, self.q_sigma)

        # vector of size latent_dim
        kl_per_latent_dim = torch.distributions.kl_divergence(q_x.to, self.prior_x).sum(axis=0)

        kl_per_point = kl_per_latent_dim.sum() / n
        kl_term = kl_per_point / q
        return kl_term


class VariationalDist(nn.Module):
    def __init__(self, num_inducing_points, batch_shape):
        super(VariationalDist, self).__init__()

        mean_init = torch.zeros(batch_shape, num_inducing_points)
        covar_init = torch.ones(batch_shape, num_inducing_points)

        self.mu = nn.Parameter(mean_init)
        self.log_sigma = nn.Parameter(covar_init)

    @property
    def sigma(self):
        return torch.nn.functional.softplus(self.log_sigma)

    def forward(self):
        # Variational distribution over the latent variable q(x)
        q_x = torch.distributions.Normal(self.mu, self.sigma)
        return q_x.rsample()

    def kl(self, prior_u):
        data_dim, n = self.mu.shape
        q_u = torch.distributions.MultivariateNormal(self.mu, torch.diag_embed(self.sigma))
        kl_per_point = torch.distributions.kl_divergence(q_u, prior_u) / n
        return kl_per_point


class CholeskeyVariationalDist(nn.Module):
    def __init__(self, num_inducing_points, batch_shape, mean_init_std=1e-3, device=None):
        super(CholeskeyVariationalDist, self).__init__()

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.batch_shape = batch_shape
        self.num_inducing_points = num_inducing_points

        mean_init = torch.zeros(num_inducing_points, device=self.device)
        covar_init = torch.eye(num_inducing_points, num_inducing_points, device=self.device)

        mean_init = mean_init.repeat(batch_shape, 1)
        covar_init = covar_init.repeat(batch_shape, 1, 1)

        self.mean_init_std = mean_init_std

        self.register_parameter(name="mu", param=torch.nn.Parameter(mean_init))
        self.register_parameter(name="chol_variational_covar", param=torch.nn.Parameter(covar_init))

    @property
    def sigma(self):
        chol_variational_covar = self.chol_variational_covar

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:], device=self.device).tril(0)
        chol_variational_covar = TriangularLinearOperator(chol_variational_covar.mul(lower_mask))

        # Now construct the actual matrix
        variational_covar = CholLinearOperator(chol_variational_covar)
        return variational_covar.evaluate()

    @property
    def covar(self):
        chol_variational_covar = self.chol_variational_covar

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:]).tril(0)
        chol_variational_covar = TriangularLinearOperator(chol_variational_covar.mul(lower_mask))

        # Now construct the actual matrix
        variational_covar = CholLinearOperator(chol_variational_covar)
        return variational_covar

    def forward(self):
        chol_variational_covar = self.chol_variational_covar
        dtype = chol_variational_covar.dtype
        device = chol_variational_covar.device

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar = TriangularLinearOperator(chol_variational_covar.mul(lower_mask))

        # Now construct the actual matrix
        variational_covar = CholLinearOperator(chol_variational_covar)
        q_x = torch.distributions.MultivariateNormal(self.mu, variational_covar)
        return q_x.rsample()

    def kl(self, prior_u):
        data_dim, n = self.mu.shape
        chol_variational_covar = self.chol_variational_covar
        dtype = chol_variational_covar.dtype
        device = chol_variational_covar.device

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:], dtype=dtype, device=device).tril(0)
        chol_variational_covar = TriangularLinearOperator(chol_variational_covar.mul(lower_mask))

        # Now construct the actual matrix
        variational_covar = CholLinearOperator(chol_variational_covar)
        q_u = torch.distributions.MultivariateNormal(self.mu, variational_covar.evaluate())
        kl_per_point = torch.distributions.kl_divergence(q_u, prior_u)
        return kl_per_point

    def shape(self):
        return torch.Size([self.batch_shape, self.num_inducing_points])

    def initialize_variational_distribution(self, prior_dist: MultivariateNormal) -> None:
        self.mu.data.copy_(prior_dist.mean)
        self.mu.data.add_(torch.randn_like(prior_dist.mean), alpha=self.mean_init_std)
        self.chol_variational_covar.data.copy_(prior_dist.lazy_covariance_matrix.cholesky().to_dense())


class SharedVariationalStrategy(nn.Module):
    def __init__(self, inducing_points, variational_distribution, jitter, device=None):
        super(SharedVariationalStrategy, self).__init__()
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.jitter = jitter
        self.m = inducing_points.shape[-2]

        ones_mat = torch.ones(variational_distribution.shape())
        result_tensors = torch.stack([torch.diag_embed(ones_mat[i]) for i in range(ones_mat.shape[0])])
        result_tensors = result_tensors.to(self.device)
        self.prior = gpytorch.distributions.MultivariateNormal(
            torch.zeros(variational_distribution.shape(), device=self.device),
            result_tensors)

    def predictive_distribution(self, k_nn, k_mm, k_mn, variational_mean, variational_cov, whitening_parameters=True):
        # k_mm = LL^T
        L = torch.linalg.cholesky(k_mm)  # torch.cholesky(k_mm, upper=False)
        m_d = variational_mean  # of size [D, M, 1]
        if len(variational_cov.shape) > 2 and variational_cov.shape[1] == variational_cov.shape[2]:
            s_d = variational_cov
        else:
            s_d = torch.diag_embed(variational_cov)  # of size [D, M, M] It's a eye matrix

        prior_dist_co = torch.eye(self.m, device=self.device)

        if whitening_parameters is True:
            # A = A=L^(-1) K_MN  (interp_term)
            interp_term = torch.linalg.solve(L, k_mn)
            # μ_f=A^T m_d^'
            # Σ_f=A^T (S'-I)A
            predictive_mean = (interp_term.transpose(-1, -2) @ m_d.unsqueeze(-1)).squeeze(-1)  # of size [D, N]
            predictive_covar = interp_term.transpose(-1, -2) @ (s_d - prior_dist_co.unsqueeze(0)) @ interp_term
        else:
            # m_f = K_NM K_MM^(-1) m_d
            # sigma_f = K_NM K_MM^(-1) (S_d-K_MM ) K_MM^(-1) K_MN
            interp_term = torch.cholesky_solve(k_mn, L, upper=False)
            predictive_mean = (interp_term.transpose(-1, -2) @ m_d.unsqueeze(-1)).squeeze(-1)  # of size [D, N]
            predictive_covar = interp_term.transpose(-1, -2) @ (s_d - k_mm) @ interp_term
        predictive_covar += k_nn
        predictive_covar += torch.eye(k_nn.shape[-1], device=self.device).squeeze(0) * self.jitter

        return torch.distributions.MultivariateNormal(predictive_mean, predictive_covar)
