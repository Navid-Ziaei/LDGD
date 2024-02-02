import numpy as np
import GPy
import torch
import matplotlib.pyplot as plt


def generate_synthetic_data(N=100, D=7, T=100, Q=2):
    # Generate latent states
    f1, f2 = 9, 4  # Example frequencies
    t = np.linspace(0, 1, T)
    x1 = np.sin(2 * np.pi * f1 * t).reshape(-1, 1)
    x2 = np.sin(2 * np.pi * f2 * t).reshape(-1, 1)
    X = np.hstack([x1, x2])  # Latent space of size TxQ

    # Randomly generate the stimulus vector for N trials
    S = torch.randint(0, 2, (N,))

    # Initialize synthetic data matrix Y of shape NxDxT
    Y = np.zeros((N, D, T))

    # Gaussian process with Matern kernel
    kernel = GPy.kern.Matern32(input_dim=Q)

    for n in range(N):
        for d in range(D):
            w1, w2 = np.random.randn(), np.random.randn()
            m_t = S[n] * x1 + (1-S[n]) * x2

            # Generate GP realization
            mu = m_t.flatten()
            C = kernel.K(X, X)
            y_nd = np.random.multivariate_normal(mu, C)

            Y[n, d, :] = y_nd

    return X, S, Y