import torch.nn.functional
from gpytorch.means import ZeroMean
from gpytorch.models.gplvm import BayesianGPLVM, VariationalLatentVariable, PointLatentVariable, MAPLatentVariable
from gpytorch.priors import NormalPrior

from torch import nn
from torch import optim
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, NaturalVariationalDistribution, TrilNaturalVariationalDistribution
from gpytorch.variational import VariationalStrategy, UnwhitenedVariationalStrategy
from gpytorch.distributions import MultivariateNormal
from src import *


class RBFKernel(nn.Module):
    def __init__(self, input_dim, variance=None, lengthscale=None, requires_grad=True):
        super(RBFKernel, self).__init__()
        self.input_dim = input_dim
        if variance is None:
            self.variance = nn.Parameter(torch.ones(1), requires_grad=requires_grad)
        else:
            self.variance = nn.Parameter(variance, requires_grad=requires_grad)

        if lengthscale is None:
            self.length_scale = nn.Parameter(torch.ones(self.input_dim), requires_grad=requires_grad)
        else:
            self.length_scale = nn.Parameter(lengthscale, requires_grad=requires_grad)

    def forward(self, x1, x2):
        # Compute squared pairwise distances
        sqdist = torch.cdist(x1, x2, p=2) ** 2
        return self.variance * torch.exp(-0.5 * sqdist / self.length_scale ** 2)


class GPRegressionModel(nn.Module):
    def __init__(self, kernel):
        super(GPRegressionModel, self).__init__()

        # Parameters
        self.kernel = kernel
        self.noise_var = nn.Parameter(torch.tensor(0.1))
        self.jitter = 1e-11

    def log_marginal_likelihood(self, x, y):
        """
        Computes the log-marginal likelihood for GP regression with added jitter for stability.

        Parameters:
            - K: Covariance matrix.
            - y: Observations.
            - noise_var: Noise variance.

        Returns:
            - Negative log-marginal likelihood.
        """
        K = self.kernel(x, x)
        n = K.size(0)
        I = torch.eye(n)

        # Add noise variance and jitter to the diagonal
        K += (self.noise_var + self.jitter) * I

        # Cholesky decomposition
        L = torch.linalg.cholesky(K, upper=False)

        # Solve for alpha
        alpha = torch.cholesky_solve(y, L)

        # Compute log-marginal likelihood
        log_likelihood = -0.5 * y.t() @ alpha - torch.sum(torch.log(torch.diag(L))) - 0.5 * n * torch.log(
            2 * torch.tensor(np.pi))
        return -log_likelihood

    def train_model(self, x, y, learning_rate=0.01, num_epochs=100, disp_interval=10):
        """Trains the GP model using the Adam optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        params = {
            'noise': [],
            'variance': [],
            'length_scale': []
        }
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.log_marginal_likelihood(x, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch % disp_interval == 0:
                print(f"Epoch {epoch}: Loss {loss.item()}")
            params['noise'].append(self.noise_var.item())
            params['variance'].append(self.kernel.variance.item())
            params['length_scale'].append(self.kernel.length_scale.item())

        # Plot the optimization process
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Negative Log-Marginal Likelihood')
        plt.title('Optimization of Negative Log-Marginal Likelihood')
        plt.grid(True)
        plt.show()

        # Plot the optimization process
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        for idx, key in enumerate(params.keys()):
            axs[idx].plot(params[key])
            axs[idx].set_xlabel('Iteration')
            axs[idx].set_ylabel(f"{key}")
            axs[idx].set_title(f'Optimization of Hyperparameters {key}')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()

        return losses

    def forward(self, x_train, y_train, x_test):
        """
        Predicts the mean and variance of the test outputs given training data.

        Args:
        - x_train (torch.Tensor): Training inputs.
        - y_train (torch.Tensor): Training outputs.
        - x_test (torch.Tensor): Test inputs.

        Returns:
        - mean (torch.Tensor): Predicted means for the test outputs.
        - var (torch.Tensor): Predicted variances for the test outputs.
        """

        # Compute kernels
        K_star = self.kernel(x_test, x_train)  # Cross-covariance matrix with training data
        K_star_star = self.kernel(x_test, x_test)  # Covariance matrix for test points
        K_train = self.kernel(x_train, x_train) + (self.noise_var + self.jitter) * torch.eye(len(x_train))

        # Compute predictive mean
        L_train = torch.cholesky(K_train, upper=False)
        alpha = torch.cholesky_solve(y_train, L_train)
        mu_star = K_star @ alpha

        # Compute predictive variance
        v = torch.cholesky_solve(K_star.t(), L_train)
        var_star = K_star_star.diag() - torch.sum(v * K_star.t(), dim=0)
        std_star = torch.sqrt(var_star).unsqueeze(1)

        return mu_star, std_star


class SparseGPRegressionModel(nn.Module):
    def __init__(self, kernel, z, m):
        super(SparseGPRegressionModel, self).__init__()

        # Parameters
        self.kernel = kernel
        self.noise_var = nn.Parameter(torch.tensor(0.1))
        self.jitter = 1e-5
        self.z = nn.Parameter(z)
        self.b = nn.Parameter(torch.zeros(m, 1))
        self.w = nn.Parameter(torch.rand(m, 1))

    def elbo(self, x, y):

        k_ff = self.kernel(x, x)
        k_uf = self.kernel(self.z, x)
        k_uu = self.kernel(self.z, self.z)

        n = k_ff.size(0)
        m = k_uu.size(0)
        I = torch.eye(n)

        # varational distribution q_phi(f) mean and
        mu_f, sigma_f = self.variational_predictive(k_ff, k_uu, k_uf, whiten=True)

        # Calculate \Q_ff = K_fu * K_uu^(-1) * K_uf
        # L L^T = Kmm + jitter I_m
        L = torch.linalg.cholesky(k_uu + self.jitter * torch.eye(m))

        # Lambda = L^{-1} Kmn
        Lambda = torch.triangular_solve(k_uf, L, upper=False)[0]

        # Lambda.t() Lambda = Knm * Knn^(-1) * Kmn
        Q_ff = Lambda.t() @ Lambda
        Q_ff += (self.noise_var ** 2 + self.jitter) * I

        mvn = gpytorch.distributions.MultivariateNormal(torch.zeros(Q_ff.shape[0]), Q_ff)

        # Compute elbo
        elbo = mvn.log_prob(y[:, 0]) - 0.5 * torch.trace(sigma_f) / (self.noise_var ** 2)

        return -elbo

    def train_model(self, x, y, learning_rate=0.01, num_epochs=100, disp_interval=10):
        """Trains the GP model using the Adam optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        params = {
            'noise': [],
            'variance': [],
            'length_scale': []
        }
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self.elbo(x, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch % disp_interval == 0:
                print(f"Epoch {epoch}: Loss {loss.item()}")
            params['noise'].append(self.noise_var.item())
            params['variance'].append(self.kernel.variance.item())
            params['length_scale'].append(self.kernel.length_scale.item())

        plot_loss(losses)

        # Plot the optimization process
        fig, axs = plt.subplots(3, 1, figsize=(10, 12))
        for idx, key in enumerate(params.keys()):
            axs[idx].plot(params[key])
            axs[idx].set_xlabel('Iteration')
            axs[idx].set_ylabel(f"{key}")
            axs[idx].set_title(f'Optimization of Hyperparameters {key}')
            axs[idx].grid(True)
        plt.tight_layout()
        plt.show()

        return losses

    def variational_predictive(self, k_ff, k_uu, k_uf, whiten=True):
        """
        Calculate the variational distribution q_phi(f)
        """
        n = k_ff.size(0)
        m = k_uu.size(0)

        # L L^T = Kmm + jitter I_m
        L = torch.linalg.cholesky(k_uu + self.jitter * torch.eye(m))

        # Lambda = L^{-1} Kmn
        Lambda = torch.triangular_solve(k_uf, L, upper=False)[0]

        # Knn - Lambda^T Lambda
        S = k_ff - torch.mm(Lambda.t(), Lambda)

        # Phi = L^{-T} L^{-1} Kmn = Kmm^{-1} Kmn
        if whiten:
            Phi = Lambda
        else:
            Phi = torch.triangular_solve(Lambda, L, upper=True, transpose=True)[0]

        # U = V^T = Phi^T W
        U = torch.mm(Phi.t(), self.w)

        # Phi^T b
        mu = torch.mm(Phi.t(), self.b)

        # S + UU^T = S + V^T V
        Sigma = S + torch.mm(U, U.t())

        return mu, Sigma

    def forward(self, x_train, y_train, x_test):
        """
        Predicts the mean and variance of the test outputs given training data.

        Args:
        - x_train (torch.Tensor): Training inputs.
        - y_train (torch.Tensor): Training outputs.
        - x_test (torch.Tensor): Test inputs.

        Returns:
        - mean (torch.Tensor): Predicted means for the test outputs.
        - var (torch.Tensor): Predicted variances for the test outputs.
        """

        # Compute kernels
        k_uu = self.kernel(self.z, self.z)
        k_ut = self.kernel(self.z, x_test)
        k_tt = self.kernel(x_test, x_test)

        mu_star, full_cov = self.variational_predictive(k_tt, k_uu, k_ut, whiten=True)
        # var_star = K_star_star.diag() - torch.sum(v * K_star.t(), dim=0)
        var_star = full_cov.diag()
        std_star = torch.sqrt(var_star).unsqueeze(1)

        return mu_star, std_star


class SparseGPClassificationModel(SparseGPRegressionModel):
    def __init__(self, kernel, z, m):
        self.whiten = True
        super(SparseGPClassificationModel, self).__init__(kernel, z, m)

    def gaussian_hermite(self, y, mean, covariance, Q):

        # Get quadrature points and weights for Gaussian-Hermite quadrature
        points, weights = np.polynomial.hermite.hermgauss(Q)
        points = torch.tensor(points)
        weights = torch.tensor(weights)

        # Transform points based on mean and covariance
        f_n = mean.t() + points[:, None] * torch.sqrt(torch.diag(covariance))

        g_n = torch.sigmoid(f_n)

        # Evaluate the Gaussian likelihood for each point
        likelihoods = y.t() * torch.log(g_n) + (1 - y.t()) * torch.log(1 - g_n)

        # Compute the weighted sum
        expected_likelihood = torch.sum(weights[:, None] * likelihoods, axis=0)

        # Normalize by the square root of pi (since the Hermite polynomials are normalized differently)
        expected_likelihood /= torch.sqrt(torch.tensor(np.pi))

        integral_approx = torch.sum(expected_likelihood)

        # Return the approximate integral
        return integral_approx

    def target_density(self, f, mu, Sigma):
        # This is q_phi(f) up to a constant
        return torch.exp(-0.5 * (f - mu).t() @ torch.inverse(Sigma) @ (f - mu))

    def proposal_sampler(self, current_state):
        # Gaussian proposal distribution
        return current_state + torch.randn_like(current_state) * 0.1

    def metropolis_hastings(self, target_density, proposal_sampler, initial_state, num_samples, burn_in=1000):
        """
        Metropolis-Hastings MCMC sampler.

        Args:
        - target_density (function): Target density to sample from (up to a constant).
        - proposal_sampler (function): Proposal distribution sampler.
        - initial_state (torch.Tensor): Initial state of the chain.
        - num_samples (int): Number of samples to draw.
        - burn_in (int): Number of initial samples to discard.

        Returns:
        - samples (list of torch.Tensor): List of MCMC samples.
        """

        current_state = initial_state
        samples = []

        for _ in range(num_samples + burn_in):
            proposed_state = proposal_sampler(current_state)
            acceptance_ratio = target_density(proposed_state) / target_density(current_state)
            if torch.rand(1) < acceptance_ratio:
                current_state = proposed_state
            samples.append(current_state)

        return samples[burn_in:]

    def metropolis_hastings_integral_approximation(self, y, mu, Sigma, num_samples, burn_in=1000):
        """
        Approximate the integral using Metropolis-Hastings MCMC.

        Args:
        - y (torch.Tensor): target values.
        - mu (torch.Tensor): mean of q_phi(f).
        - Sigma (torch.Tensor): covariance of q_phi(f).
        - num_samples (int): number of MCMC samples.
        - burn_in (int): burn-in samples to discard.

        Returns:
        - integral_approximation (float): the approximate value of the integral.
        """

        # Use Metropolis-Hastings to sample from q_phi(f)
        initial_state = mu
        samples = self.metropolis_hastings(lambda f: self.target_density(f, mu, Sigma),
                                           self.proposal_sampler, initial_state, num_samples, burn_in)

        # Approximate the integral using the MCMC samples
        function_values = [self.log_liklihood(sample, y) for sample in samples]
        integral_approximation = torch.mean(torch.tensor(function_values))

        return integral_approximation.item()

    def log_liklihood(self, f, y):
        probs = torch.sigmoid(f)
        log_likelihood = y * torch.log(probs) + (1 - y) * torch.log(1 - probs)
        return torch.sum(log_likelihood)

    def kl_divergence_gaussians(self, mu_0, Sigma_0, mu_1, Sigma_1):
        """
        Computes the KL divergence between two multivariate Gaussians.

        Args:
        - mu_0, mu_1 (torch.Tensor): Mean vectors of the Gaussians.
        - Sigma_0, Sigma_1 (torch.Tensor): Covariance matrices of the Gaussians.

        Returns:
        - KL divergence (torch.Tensor)
        """
        k = mu_0.shape[0]

        Sigma_1_inv = torch.inverse(Sigma_1)
        mu_diff = mu_1 - mu_0

        term1 = torch.logdet(Sigma_1) - torch.logdet(Sigma_0 + self.jitter * torch.eye(k))
        term2 = torch.trace(torch.mm(Sigma_1_inv, Sigma_0))
        term3 = torch.mm(torch.mm(mu_diff.t(), Sigma_1_inv), mu_diff)

        kl = 0.5 * (term1 - k + term2 + term3)

        return kl

    def elbo(self, x, y):
        # This remains largely the same, but the likelihood term changes
        k_ff = self.kernel(x, x)
        k_uf = self.kernel(self.z, x)
        k_uu = self.kernel(self.z, self.z)

        m = k_uu.size(0)

        # calculate q(f)
        mu_f, sigma_f = self.variational_predictive(k_ff, k_uu, k_uf)

        # Calculate E_q(f)[log p(y|f)]
        expected_log_likelihood = self.gaussian_hermite(y, mean=mu_f, covariance=sigma_f, Q=100)
        # expected_log_likelihood = self.metropolis_hastings_integral_approximation(y, mu=mu_f, Sigma=sigma_f,
        #                                                                           num_samples=1000, burn_in=1000)

        # KL divergence remains the same
        if self.whiten is True:
            Sigma_1_whitened = torch.eye(m)
            kl_term = self.kl_divergence_gaussians(self.b, self.w @ self.w.t(), torch.zeros(m, 1), Sigma_1_whitened)
        else:
            kl_term = self.kl_divergence_gaussians(self.b, self.w @ self.w.t(), torch.zeros(m, 1), k_uu)

        elbo = expected_log_likelihood - kl_term
        return -elbo

    # forward method remains largely the same, but you might return probabilities
    def forward(self, x_train, y_train, x_test):
        mu_star, std_star = super().forward(x_train, y_train, x_test)
        return torch.sigmoid(mu_star), std_star


class GPLVM_point_estimate(nn.Module):
    def __init__(self, y, kernel, q=2):
        super(GPLVM_point_estimate, self).__init__()

        # Parameters
        self.kernel = kernel
        self.jitter = 1e-6

        self.n = y.shape[0]
        self.d = y.shape[1]
        self.q = q
        self.x = nn.Parameter(torch.randn(self.n, self.q))
        self.noise_var = nn.Parameter(torch.tensor(0.1))

    def log_likelihood(self, y):
        log_likelihoods = []
        K = self.kernel(self.x, self.x)
        n = K.size(0)
        I = torch.eye(n)

        # Add noise variance and jitter to the diagonal
        K += (self.noise_var + self.jitter) * I

        for d in range(self.d):
            y_d = y[:, d]

            # Multivariate normal with zero mean
            mvn = MultivariateNormal(torch.zeros(self.n), K)
            log_likelihoods.append(mvn.log_prob(y_d))

        return torch.sum(torch.stack(log_likelihoods))

    def forward(self, x_star):
        # Since this is a latent model, the forward pass would require the prediction mechanism for GPLVM.
        # This is just a simple example to compute the mean for given inputs x_star
        K_xx = self.kernel(self.x, self.x) + self.jitter * torch.eye(self.n)
        K_xs = self.kernel(self.x, x_star)
        K_ss = self.kernel(x_star, x_star)

        K_xx_inv = torch.inverse(K_xx)

        mean = torch.mm(torch.mm(K_xs.t(), K_xx_inv), self.y)

        return mean

    def train_model(self, y, learning_rate=0.01, epochs=100):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = -self.log_likelihood(y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        plot_loss(losses)

        return losses


class JointGPLVM_PointEstimate(nn.Module):
    def __init__(self, y, kernel, q=2):
        super(JointGPLVM_PointEstimate, self).__init__()

        # Parameters
        self.kernel = kernel
        self.jitter = 1e-6

        self.n = y.shape[0]
        self.d = y.shape[1]
        self.q = q
        self.x = nn.Parameter(torch.randn(self.n, self.q))
        self.noise_var = nn.Parameter(torch.tensor(0.1))

    def log_likelihood(self, y, s):
        log_likelihoods = []
        K = self.kernel(self.x, self.x)
        n = K.size(0)
        I = torch.eye(n)

        # Add noise variance and jitter to the diagonal
        K += (self.noise_var + self.jitter) * I

        for d in range(self.d):
            y_d = y[:, d]

            # Multivariate normal with zero mean
            mvn = MultivariateNormal(torch.zeros(self.n), K)
            log_likelihoods.append(mvn.log_prob(y_d))

        return torch.sum(torch.stack(log_likelihoods))

    def forward(self, x_star):
        # Since this is a latent model, the forward pass would require the prediction mechanism for GPLVM.
        # This is just a simple example to compute the mean for given inputs x_star
        K_xx = self.kernel(self.x, self.x) + self.jitter * torch.eye(self.n)
        K_xs = self.kernel(self.x, x_star)
        K_ss = self.kernel(x_star, x_star)

        K_xx_inv = torch.inverse(K_xx)

        mean = torch.mm(torch.mm(K_xs.t(), K_xx_inv), self.y)

        return mean

    def train_model(self, y, learning_rate=0.01, epochs=100):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            loss = -self.log_likelihood(y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
        plot_loss(losses)

        return losses


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_model(self, x_train, y_train, likelihood, training_iter=100):
        self.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, self)
        losses = []

        for i in range(training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self(x_train)
            # Calc loss and backprop gradients
            loss = -mll(output, y_train)
            loss.backward()
            losses.append(loss.item())
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.covar_module.base_kernel.lengthscale.item(),
                self.likelihood.noise.item()
            ))
            optimizer.step()
        return losses


class SPGPModelGPy(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution,
                                                   learn_inducing_locations=True)
        super(SPGPModelGPy, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def train_model(self, x_train, y_train, likelihood, training_iter=100):
        self.train()
        likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.parameters()},
            {'params': likelihood.parameters()},
        ], lr=0.01)

        # Our loss object. We're using the VariationalELBO
        mll = gpytorch.mlls.VariationalELBO(likelihood, self, num_data=y_train.size(0))
        losses = []
        inducing_points_history = []

        for i in range(training_iter):
            # Within each iteration, we will go over each minibatch of data
            optimizer.zero_grad()
            output = self(x_train)
            loss = -mll(output, y_train)
            losses.append(loss.item())
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                self.covar_module.base_kernel.lengthscale.item(),
                likelihood.noise.item()
            ))
            loss.backward()
            optimizer.step()
            # Save inducing point locations for animation
            inducing_points_history.append(self.variational_strategy.inducing_points.detach().clone())
        plot_loss(losses)
        # create_animation(x_train, y_train, inducing_points_history)

        return losses, likelihood


class GPClassificationModel(ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = UnwhitenedVariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return latent_pred

    def train_model(self, train_x, train_y, likelihood, training_iterations=1000):
        # Find optimal model hyperparameters
        self.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        # num_data refers to the number of training datapoints
        mll = gpytorch.mlls.VariationalELBO(likelihood, self, train_y.numel())
        losses = []
        for i in range(training_iterations):
            # Zero backpropped gradients from previous iteration
            optimizer.zero_grad()
            # Get predictive output
            output = self(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            losses.append(loss.item())
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
        plot_loss(losses)
        return losses, likelihood


def _init_pca(Y, latent_dim):
    U, S, V = torch.pca_lowrank(Y, q=latent_dim)
    return torch.nn.Parameter(torch.matmul(Y, V[:, :latent_dim]))


class bGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, latent_dim, n_inducing, pca=False, mode='bayesian', data=None):
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
            X_init = _init_pca(data, latent_dim)  # Initialise X to PCA
        else:
            X_init = torch.nn.Parameter(torch.randn(n, latent_dim))

        # LatentVariable (c)
        if mode.lower() == 'point_estimate':
            X = PointLatentVariable(n, latent_dim, X_init)
        elif mode.lower() == 'bayesian':
            X = VariationalLatentVariable(n, data_dim, latent_dim, X_init, prior_x)
        elif mode.lower() == 'map':
            X = MAPLatentVariable(n, latent_dim, X_init, prior_x)
        else:
            raise ValueError("The mode should be from ['Point_estimate', 'Bayesian', 'MAP']")

        # For (a) or (b) change to below:
        # X = PointLatentVariable(n, latent_dim, X_init)
        # X = MAPLatentVariable(n, latent_dim, X_init, prior_x)

        super().__init__(X, q_f)

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

    def train_model(self, Y, optimizer, mll, epochs=5000):
        loss_list = []
        iterator = range(epochs)
        batch_size = 100
        list_alpha = []
        for i in iterator:
            batch_index = self._get_batch_idx(batch_size)
            optimizer.zero_grad()
            sample = self.sample_latent_variable()  # a full sample returns latent x across all N
            sample_batch = sample[batch_index]
            output_batch = self(sample_batch)
            loss = -mll(output_batch, Y[batch_index].T).sum()
            loss_list.append(loss.item())
            print('Loss: ' + str(float(np.round(loss.item(), 2))) + ", iter no: " + str(i))
            loss.backward()
            optimizer.step()
            list_alpha.append(self.covar_module.base_kernel.lengthscale.detach().numpy())

        plot_loss(loss_list)
        plot_loss(1/np.stack(list_alpha).squeeze())
        return loss_list


class shared_bGPLVM(BayesianGPLVM):
    def __init__(self, n, data_dim, num_classes, latent_dim, n_inducing, pca=False):
        self.n = n
        self.batch_shape_reg = torch.Size([data_dim])
        self.batch_shape_cls = torch.Size([num_classes])

        # Locations Z_{d} corresponding to u_{d}, they can be randomly initialized or
        # regularly placed with shape (D x n_inducing x latent_dim).
        self.inducing_inputs = torch.randn(data_dim, n_inducing, latent_dim)

        # Sparse Variational Formulation (inducing variables initialised as randn)
        q_u_n = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape_reg)
        q_f_n = VariationalStrategy(self, self.inducing_inputs, q_u_n, learn_inducing_locations=True)

        q_u_s = CholeskyVariationalDistribution(n_inducing, batch_shape=self.batch_shape_cls)
        q_f_s = VariationalStrategy(self, self.inducing_inputs, q_u_s, learn_inducing_locations=True)

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

        super().__init__(X, q_f_n)

        # Kernel (acting on latent dimensions)
        self.mean_module = ZeroMean(ard_num_dims=latent_dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist

    def _get_batch_idx(self, batch_size):
        valid_indices = np.arange(self.n)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

    def train_model(self, Y, optimizer, mll):
        loss_list = []
        iterator = range(10000)
        batch_size = 100
        for i in iterator:
            batch_index = self._get_batch_idx(batch_size)
            optimizer.zero_grad()
            sample = self.sample_latent_variable()  # a full sample returns latent x across all N
            sample_batch = sample[batch_index]
            output_batch = self(sample_batch)
            loss = -mll(output_batch, Y[batch_index].T).sum()
            loss_list.append(loss.item())
            print('Loss: ' + str(float(np.round(loss.item(), 2))) + ", iter no: " + str(i))
            loss.backward()
            optimizer.step()
        plot_loss(loss_list)
        return loss_list
