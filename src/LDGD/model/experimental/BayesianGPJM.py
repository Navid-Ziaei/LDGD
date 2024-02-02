import torch
import torch.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# Set up PyTorch
torch.manual_seed(0)  # for reproducibility


# RBF Kernel function using PyTorch
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


class BayesianGPJM(nn.Module):
    def __init__(self, T, Q):
        super(BayesianGPJM, self).__init__()
        # Parameters for the variational posterior q_phi(X)
        # We assume a Gaussian distribution with diagonal covariance
        self.means = nn.Parameter(torch.randn(Q, T))
        self.log_vars = nn.Parameter(torch.randn(Q, T))

        # Parameters for the RBF kernels
        self.phi = nn.Parameter(torch.randn(1))
        self.psi = nn.Parameter(torch.randn(1))

        # define kernels
        self.kernel_x = RBFKernel(1, variance=torch.tensor(1.0), lengthscale=torch.randn(1), requires_grad=True)
        self.kernel_y = RBFKernel(1, variance=torch.tensor(1.0), lengthscale=torch.randn(1), requires_grad=True)
        self.kernel_s = RBFKernel(1, variance=torch.tensor(1.0), lengthscale=torch.randn(1), requires_grad=True)

        self.jitter = 1e-4

    def calculate_elbo(self, Y, S):
        # Sample from q_phi(X) using the reparameterization trick
        X_sample = self.means + torch.exp(0.5 * self.log_vars) * torch.randn_like(self.means)

        # Compute the expected joint log likelihood under q_phi(X)

        cov_y = self.kernel_y(X_sample, X_sample)
        cov_s = self.kernel_s(X_sample, X_sample)
        
        cov_y += torch.eye(cov_y.shape[0]) * self.jitter
        cov_s += torch.eye(cov_s.shape[0]) * self.jitter

        # L_y = torch.linalg.cholesky(cov_y)
        # L_s = torch.linalg.cholesky(cov_s)

        m_y = torch.zeros((cov_y.shape[0], 1))
        m_s = torch.zeros((cov_s.shape[0], 1))

        log_likelihood_Y = torch.sum(dist.MultivariateNormal(m_y.squeeze(), cov_y).log_prob(Y.t()))
        log_likelihood_S = torch.sum(dist.MultivariateNormal(m_s.squeeze(), cov_s).log_prob(S.t()))

        # Compute the entropy of q_phi(X) (negative because we want to maximize ELBO)
        entropy = -0.5 * torch.sum(self.log_vars + np.log(2 * np.pi) + 1)

        # Compute the ELBO
        ELBO = log_likelihood_Y + log_likelihood_S + entropy

        return -ELBO

    def forward(self, Y, S):
        pass

    def train(self, Y, S, optimizer, num_epochs=2000, disp_interval=10):
        # Training loop for VI
        loss_history_vi = []
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            negative_ELBO = self.calculate_elbo(Y, S)
            negative_ELBO.backward(retain_graph=True)
            optimizer.step()

            # Store the loss value
            loss_history_vi.append(negative_ELBO.item())

            # Print training info
            if (epoch + 1) % disp_interval == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Negative ELBO: {negative_ELBO.item():.4f}")

        # Plot the training loss for VI
        plt.plot(loss_history_vi)
        plt.xlabel('Epoch')
        plt.ylabel('Negative ELBO')
        plt.title('Training Loss (Variational Inference)')
        plt.show()


def generate_synthetic_data(T, N, jitter, frequencies, sigma_true_x, sigma_true_s, sigma_true_y):
    # Generate latent variable X (two sinusoids) with PyTorch
    t_tensor = torch.linspace(0, 1.5, T)[:, None]
    X_tensor = torch.stack([torch.squeeze(torch.sin(2 * np.pi * freq * t_tensor)) for freq in frequencies])
    X_tensor = X_tensor.t()

    # define kernels
    kernel_x = RBFKernel(1, variance=torch.tensor(1.0), lengthscale=torch.tensor(sigma_true_x), requires_grad=False)
    kernel_y = RBFKernel(1, variance=torch.tensor(1.0), lengthscale=torch.tensor(sigma_true_y), requires_grad=False)
    kernel_s = RBFKernel(1, variance=torch.tensor(1.0), lengthscale=torch.tensor(sigma_true_s), requires_grad=False)

    cov_x = kernel_x(t_tensor, t_tensor)
    cov_y = kernel_y(X_tensor, X_tensor)
    cov_s = kernel_s(X_tensor, X_tensor)

    cov_x += torch.eye(cov_x.shape[0]) * jitter
    cov_y += torch.eye(cov_y.shape[0]) * jitter
    cov_s += torch.eye(cov_s.shape[0]) * jitter

    # Compute Cholesky decomposition
    L_x = torch.linalg.cholesky(cov_x)
    L_y = torch.linalg.cholesky(cov_y)
    L_s = torch.linalg.cholesky(cov_s)

    # define mean
    m_x = torch.zeros((cov_x.shape[0], 1))
    m_y = torch.zeros((cov_y.shape[0], 1))
    m_s = torch.zeros((cov_s.shape[0], 1))

    # Sample z from standard normal distribution
    z_x = torch.randn(m_x.shape[0], N)
    z_y = torch.randn(m_y.shape[0], N)
    z_s = torch.randn(m_s.shape[0], N)

    # Transform the sample
    x_true = m_x + torch.matmul(L_x, z_x)
    y = m_y + torch.matmul(L_y, z_y)
    s = m_s + torch.matmul(L_s, z_s)

    y = y.detach()
    s = s.detach()

    print(f"X shape is {X_tensor.shape}")
    print(f"Y shape is {y.shape}")
    print(f"S shape is {s.shape}")

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.subplot(3, 1, 1)
    plt.plot(t_tensor, X_tensor[:, 0].numpy(), label="Sinusoid 1")
    plt.plot(t_tensor, X_tensor[:, 1].numpy(), label="Sinusoid 2")
    plt.title("Latent Variable X")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t_tensor, y.detach().numpy())
    plt.title("Generated Neural Data $Y_i$")

    plt.subplot(3, 1, 3)
    plt.plot(t_tensor, s.detach().numpy())
    plt.title("Generated Label $S_i$")
    plt.tight_layout()

    plt.show()

    return X_tensor, y, s