from src import *

# Parameters
T = 25  # 1.5 seconds, assuming 1ms per sample
Q = 2
N = 100  # Number of data samples (can be increased if multiple samples are required)
frequencies = [5, 10]  # Frequencies for the sinusoids
sigma_true_x = 0.4  # Hyperparameter for the RBF kernel
sigma_true_y = 0.2  # Hyperparameter for the RBF kernel
sigma_true_s = 1.0  # Hyperparameter for the RBF kernel

jitter = 1e-4

X_tensor, y, s = generate_synthetic_data(T, N, jitter, frequencies, sigma_true_x, sigma_true_s, sigma_true_y)

# Instantiate the variational model and optimizer
variational_model = BayesianGPJM(T, Q)
optimizer = optim.Adam(variational_model.parameters(), lr=0.01)
variational_model.train(y, s, optimizer=optimizer, num_epochs=50000, disp_interval=1000)
