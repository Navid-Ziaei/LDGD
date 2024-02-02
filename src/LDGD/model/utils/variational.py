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

class VariationalLatentVariable(nn.Module):
    def __init__(self, n, data_dim, latent_dim, X_init, prior_x):
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
        self.kl_loss_list.append(self.kl_loss.detach().numpy())
        return q_x.rsample()

    def kl(self):
        n, q = self.q_mu.shape
        q_x = torch.distributions.Normal(self.q_mu, self.q_sigma)
        kl_per_latent_dim = torch.distributions.kl_divergence(q_x, self.prior_x).sum(
            axis=0)  # vector of size latent_dim
        kl_per_point = kl_per_latent_dim.sum() / n  # scalar
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
    def __init__(self, num_inducing_points, batch_shape):
        super(CholeskeyVariationalDist, self).__init__()

        mean_init = torch.zeros(num_inducing_points)
        covar_init = torch.eye(num_inducing_points, num_inducing_points)

        mean_init = mean_init.repeat(batch_shape, 1)
        covar_init = covar_init.repeat(batch_shape, 1, 1)

        self.register_parameter(name="mu", param=torch.nn.Parameter(mean_init))
        self.register_parameter(name="chol_variational_covar", param=torch.nn.Parameter(covar_init))

    @property
    def sigma(self):
        chol_variational_covar = self.chol_variational_covar

        # First make the cholesky factor is upper triangular
        lower_mask = torch.ones(self.chol_variational_covar.shape[-2:]).tril(0)
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
        kl_per_point = torch.distributions.kl_divergence(q_u, prior_u) / n
        return kl_per_point

class DoublyVariationalStrategy(VariationalStrategy):
    def __init__(self):
        super(DoublyVariationalStrategy, self).__init__()

    def forward(
        self,
        x: Tensor,
        inducing_points: Tensor,
        inducing_values: Tensor,
        variational_inducing_covar = None,
        **kwargs,
    ) -> MultivariateNormal:
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(self.jitter_val)
        induc_data_covar = full_covar[..., :num_induc, num_induc:].to_dense()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(self.jitter_val).to_dense()
                + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)