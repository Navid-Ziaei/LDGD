from src import *
from src import *

# Set the seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

""" ========================================================================= """
"""                             Classification                                """
""" ========================================================================= """
x_train, y_train, x_test, y_test = generate_synthetic_classification_data(N=50)
m = 10
inducing_points = torch.linspace(0.5, 4.5, m)[:, None]

""" ==================== Classification from Scratch ======================= """

kernel = RBFKernel(input_dim=1)
model = SparseGPClassificationModel(kernel=kernel, z=inducing_points, m=m)

model.train_model(x_train, y_train, num_epochs=10000)
mu_star, std_test = model(x_train, y_train, x_test)

plot_with_confidence(x_train, y_train, x_test, mu_star, torch.sqrt(mu_star*(1-mu_star)),
                     inducing_points=model.z.detach().numpy())


""" ==================== Classification using GPyTorch ======================= """
# Initialize model and likelihood

"""
model = GPClassificationModel(x_train[:, 0])
likelihood = gpytorch.likelihoods.BernoulliLikelihood()
losses, likelihood = model.train_model(x_train[:, 0], y_train[:, 0], likelihood, training_iterations=1000)
model.eval()
likelihood.eval()

f_preds = model(x_test[:, 0])
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    y_preds = likelihood(model(x_test[:, 0]))

y_mean = y_preds.mean
y_std = np.sqrt(y_preds.variance)

plot_with_confidence(x_train, y_train, x_test, f_preds.mean, np.sqrt(f_preds.variance.detach()))
plot_with_confidence(x_train, y_train, x_test, y_mean, y_std)
"""
