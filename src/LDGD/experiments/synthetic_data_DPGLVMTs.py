import GPy
from gp_project_pytorch.model import *
# Constants
N = 100
D = 10
Q = 2
variance = 1
length_scale = 2
sigma1 = 0.1
sigma2 = 0.2
mu1 = np.array([0, 1])
mu2 = np.array([-1, -1])

# Generate S
S = np.random.binomial(1, 0.5, N)  # Assuming equal probability for two classes


# Define Squared Exponential Kernel
kernel = GPy.kern.RBF(input_dim=Q, variance=1.0, lengthscale=length_scale)

# Generate X based on S
X = np.zeros((N, Q))
for i, s in enumerate(S):
    if s:
        cov = sigma1 * np.eye(Q)
        X[i, :] = np.random.multivariate_normal(mu1, cov)
    else:
        cov = sigma2 * np.eye(Q)
        X[i, :] = np.random.multivariate_normal(mu2, cov)

# Generate Y based on X
Y = np.zeros((N, D))
K = kernel.K(X,X)
for d in range(D):
    Y[:, d] = np.random.multivariate_normal(np.zeros(N), K)

print(X)
print(Y)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=S)
plt.show()

kernel = GPy.kern.RBF(input_dim=Q, variance=1.0, lengthscale=1.0)
model = DGPLVM(Q=2, kernel=kernel)
