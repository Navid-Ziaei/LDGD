import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal
import numpy as np
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, Normal
from torch import distributions

import matplotlib.pyplot as plt


def plot_result(losses, param_history, true_params=None):
    fig, axs = plt.subplots(5, 1, figsize=(6, 10))

    # Plot Loss
    axs[0].plot(losses, label='Loss')
    axs[0].set_title('Loss over Time')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    axs[0].legend()

    # Plot lengthscale_s
    axs[1].plot(param_history['lengthscale_s'], label='Estimated lengthscale_s')
    if true_params is not None:
        axs[1].axhline(y=true_params['true_s_lengthscale'], color='r', linestyle='--', label=f'True lengthscale_s = {true_params["true_s_lengthscale"]}')
    axs[1].set_title('Lengthscale_s over Time')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel('Lengthscale_s')
    axs[1].grid(True)
    axs[1].legend()

    # Plot variance_s
    axs[2].plot(param_history['variance_s'], label='Estimated variance_s')
    if true_params is not None:
        axs[2].axhline(y=true_params['true_s_variance'], color='r', linestyle='--', label=f'True variance_s = {true_params["true_s_variance"]}')
    axs[2].set_title('Variance_s over Time')
    axs[2].set_xlabel('Iterations')
    axs[2].set_ylabel('Variance_s')
    axs[2].grid(True)
    axs[2].legend()

    # Plot lengthscale_x
    axs[3].plot(param_history['lengthscale_x'], label='Estimated lengthscale_x')
    if true_params is not None:
        axs[3].axhline(y=true_params["true_x_lengthscale"], color='r', linestyle='--', label=f'True lengthscale_x = {true_params["true_x_lengthscale"]}')
    axs[3].set_title('Lengthscale_x over Time')
    axs[3].set_xlabel('Iterations')
    axs[3].set_ylabel('Lengthscale_x')
    axs[3].grid(True)
    axs[3].legend()

    # Plot variance_x
    axs[4].plot(param_history['variance_x'], label='Estimated variance_x')
    if true_params is not None:
        axs[4].axhline(y=true_params["true_x_variance"], color='r', linestyle='--', label=f'True variance_x = {true_params["true_x_variance"]}')
    axs[4].set_title('Variance_x over Time')
    axs[4].set_xlabel('Iterations')
    axs[4].set_ylabel('Variance_x')
    axs[4].grid(True)
    axs[4].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


class RBFKernel(nn.Module):
    def __init__(self, input_dim, variance=None, lengthscale=None, requires_grad=True):
        super(RBFKernel, self).__init__()
        self.input_dim = input_dim
        if variance is None:
            self.variance = nn.Parameter(torch.ones(1), requires_grad=requires_grad)
        else:
            self.variance = nn.Parameter(variance, requires_grad=requires_grad)

        if lengthscale is None:
            self.lengthscale = nn.Parameter(torch.ones(self.input_dim), requires_grad=requires_grad)
        else:
            self.lengthscale = nn.Parameter(lengthscale, requires_grad=requires_grad)

    def forward(self, x1, x2):
        # Compute squared Euclidean distance
        dist = torch.cdist(x1, x2, p=2) ** 2
        cov = self.variance * torch.exp(-0.5 * dist / (self.lengthscale ** 2))
        return cov


class DGPLVM_v1(nn.Module):
    """
    The distributions governing these relationships are:
    \begin{align*}
        Y_{:,d}|X &\sim \mathcal{N}(0, K_{\theta}(X,X')) \\
        X_i|S_i &\sim \mathcal{N}(0, \sigma_1^{S_i} \sigma_2^{(1-S_i)}{I}_Q)
    \end{align*}


    Given these distributions, we can express the probability of observing \(Y\) given the labels \(S\) as the marginal likelihood:
    \begin{align}
        P(Y|S) &= \int P(Y|X,S) P(X|S) \, dX \\
        &= \int P(Y|X) P(X|S) \, dX \\
    \end{align}
    """

    def __init__(self, N, Q, D, kernel_S, kernel_x, jitter=1e-7):
        super(DGPLVM_v1, self).__init__()

        # Hyperparameters
        self.N = N
        self.Q = Q
        self.D = D

        # Kernels
        self.kernel_S = kernel_S
        self.kernel_X = kernel_x

        self.jitter = jitter

    def sample_X_given_S(self, S, n_sample=1):
        dist = MultivariateNormal(torch.zeros(self.N),
                                  covariance_matrix=self.kernel_S(S, S) + self.jitter * torch.eye(self.N))
        samples = dist.sample_n(n_sample)
        log_prob = dist.log_prob(samples)
        return samples, log_prob

    def log_likelihood(self, Y, X, X_logprob):
        batch_size = X.shape[0]
        K = self.kernel_X(X, X) + self.jitter * torch.eye(self.N)
        # log_probs = self.compute_log_Y_given_X(Y, K)
        # Reshape for batched MultivariateNormal, but check the dimension of K first
        if len(K.shape) == 2:  # if K is 2D
            K_batched = K.unsqueeze(0).repeat(batch_size, 1, 1)
        else:  # if K is already 3D
            K_batched = K

        dist = distributions.MultivariateNormal(torch.zeros(self.N).repeat(batch_size, 1), covariance_matrix=K_batched)

        # Compute log probability for all features at once
        log_probs = 0.0
        for d in range(Y.shape[-1]):
            log_probs += dist.log_prob(Y[:, :, d])
        return log_probs

    def calculate_log_likelihood_for_samples(self, Y, S, X_samples, X_samples_log_prob):
        # Stack X samples
        if len(X_samples.shape) == 2:
            X_samples = X_samples[:, :, None]

        # Expand Y to match the number of samples for batch processing
        repeated_Y = Y.unsqueeze(0).repeat(X_samples.shape[0], 1, 1)

        # Compute log likelihood for all samples at once
        log_likelihoods = self.log_likelihood(repeated_Y, X_samples, X_samples_log_prob)

        return -torch.mean(log_likelihoods)

    def maximize_likelihood(self, Y, S, num_iterations=100, num_samples=10, lr=0.01, true_params=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        param_history = {
            "lengthscale_s": [],
            "variance_s": [],
            "lengthscale_x": [],
            "variance_x": [],
        }
        X_samples, X_samples_log_prob = self.sample_X_given_S(S, num_samples)
        for idx in range(num_iterations):
            try:
                loss = self.calculate_log_likelihood_for_samples(Y, S, X_samples, X_samples_log_prob)
            except:
                plot_result(losses, param_history, true_params=true_params)
                break
            losses.append(loss.item())

            # Store parameter values
            param_history["lengthscale_s"].append(self.kernel_S.lengthscale.item())
            param_history["variance_s"].append(self.kernel_S.variance.item())
            param_history["lengthscale_x"].append(self.kernel_X.lengthscale.item())
            param_history["variance_x"].append(self.kernel_X.variance.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 5 == 0:
                print(f"Iteration {idx}: \t Loss: {losses[-1]}")

        return losses, param_history


class DGPLVM_v2(DGPLVM_v1):
    def __init__(self, N, Q, D, kernel_S, kernel_x, jitter=1e-7, min_lengthscale=0.1):
        super(DGPLVM_v2, self).__init__(N, Q, D, kernel_S, kernel_x, jitter)
        self.min_lengthscale = min_lengthscale
        self.reg_lambda = 0.01

    def maximize_likelihood(self, Y, S, num_iterations=100, num_samples=10, lr=0.01, true_params=None):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        losses = []
        param_history = {
            "lengthscale_s": [],
            "variance_s": [],
            "lengthscale_x": [],
            "variance_x": [],
        }
        X_samples, X_samples_log_prob = self.sample_X_given_S(S, num_samples)
        for idx in range(num_iterations):
            try:
                # Introduce a regularization term
                reg_term = self.reg_lambda * (self.kernel_X.variance.pow(2) + self.kernel_X.lengthscale.pow(2))
                loss = self.calculate_log_likelihood_for_samples(Y, S, X_samples, X_samples_log_prob) + reg_term
            except:
                plot_result(losses, param_history, true_params=true_params)
                break
            losses.append(loss.item())

            param_history["lengthscale_s"].append(self.kernel_S.lengthscale.item())
            param_history["variance_s"].append(self.kernel_S.variance.item())
            param_history["lengthscale_x"].append(self.kernel_X.lengthscale.item())
            param_history["variance_x"].append(self.kernel_X.variance.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Enforce constraints on lengthscale_x after update
            self.kernel_X.lengthscale.data = torch.clamp(self.kernel_X.lengthscale.data, min=self.min_lengthscale)

            if idx % 5 == 0:
                print(f"Iteration {idx}: \t Loss: {losses[-1]}")

        return losses, param_history



def generate_data(N, D, Q, true_s_variance=0.1, true_s_lengthscale=0.1, true_x_lengthscale=1.5, true_x_variance=2.5,
                  jitter=1e-6):
    # Sample S ~ Binomial(0.5)
    # S = torch.randint(0, 2, (N, 1)).float()
    S = torch.linspace(0, 2, N)[:, None].float()

    # Define kernels
    kernel_S = RBFKernel(N, variance=torch.ones(1) * true_s_variance,
                         lengthscale=torch.ones(1) * true_s_lengthscale,
                         requires_grad=False)
    kernel_X = RBFKernel(N, variance=torch.ones(1) * true_x_variance,
                         lengthscale=torch.ones(1) * true_x_lengthscale)

    # Constructing a new covariance matrix for X using a Gaussian kernel based on S values
    S_cov = kernel_S(S, S)
    S_cov += jitter * torch.eye(N)

    fig, axs = plt.subplots(1, 2, figsize=(12, 12))
    axs[0].imshow(S_cov)
    axs[0].set_title("$K(S,S)$")
    # Sample X given the adjusted covariance matrix
    true_mu = torch.zeros(N)
    dist_X = MultivariateNormal(true_mu, S_cov)
    X = dist_X.sample().unsqueeze(1)

    # Sample Y given X using Gaussian Process
    X_cov = kernel_X(X, X)
    X_cov += jitter * torch.eye(N)

    axs[1].imshow(X_cov.detach().numpy())
    axs[1].set_title("$K(X,X)$")

    dist_Y = MultivariateNormal(torch.zeros(N), covariance_matrix=X_cov)
    Y_obs = torch.stack([dist_Y.sample() for _ in range(D)], dim=1)

    return Y_obs, X, S
