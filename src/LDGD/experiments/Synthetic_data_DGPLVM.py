import matplotlib.pyplot as plt

from gp_project_pytorch.model import *

N, D, Q = 5, 5, 1

true_x_lengthscale = 0.4
true_x_variance = 0.4
true_s_lengthscale = 0.4
true_s_variance = 0.4

true_params = {
    'true_x_lengthscale': true_x_lengthscale,
    'true_x_variance': true_x_variance,
    'true_s_lengthscale': true_s_lengthscale,
    'true_s_variance': true_s_variance
}
# Generate synthetic
Y, X, S = generate_data(N, D, Q,
                        true_s_variance=true_s_variance,
                        true_s_lengthscale=true_s_lengthscale,
                        true_x_lengthscale=true_x_lengthscale,
                        true_x_variance=true_x_variance,
                        jitter=1e-6)

kernel_x = RBFKernel(N, variance=torch.ones(1) * 0.3,
                     lengthscale=torch.ones(1) * 0.3)
kernel_s = RBFKernel(N, variance=torch.ones(1) * true_s_variance,
                     lengthscale=torch.ones(1) * true_s_lengthscale,
                     requires_grad=True)

model = DGPLVM_v2(N, Q, D, kernel_s, kernel_x, jitter=1e-6)
losses, param_history = model.maximize_likelihood(Y, S,
                                                  num_iterations=500,
                                                  num_samples=4000,
                                                  lr=0.1,
                                                  true_params=true_params)
plot_result(losses, param_history, true_params)

print(f"predicted lengthscale {param_history['lengthscale_x'][-1]} \t true lengthscale {true_x_lengthscale}")
print(f"predicted variance {param_history['variance_x'][-1]} \t true variance {true_x_variance}")


optimizer_lbfgs_v2 = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20, history_size=10)

def closure_v2():
    optimizer_lbfgs_v2.zero_grad()
    X_samples, X_samples_log_prob = model.sample_X_given_S(S, 4000)
    loss = model.calculate_log_likelihood_for_samples(Y, S, X_samples, X_samples_log_prob)
    loss.backward()
    return loss

losses_lbfgs_v3 = []
param_history_lbfgs_v3 = {
    "lengthscale_s": [],
    "variance_s": [],
    "lengthscale_x": [],
    "variance_x": [],
}

# Optimization using L-BFGS with the modified model and closure
for idx in range(500):
    optimizer_lbfgs_v2.step(closure_v2)

    # Store loss and parameter values
    with torch.no_grad():
        X_samples, X_samples_log_prob = model.sample_X_given_S(S, 4000)
        loss = model.calculate_log_likelihood_for_samples(Y, S, X_samples, X_samples_log_prob)
        losses_lbfgs_v3.append(loss.item())

        param_history_lbfgs_v3["lengthscale_s"].append(model.kernel_S.lengthscale.item())
        param_history_lbfgs_v3["variance_s"].append(model.kernel_S.variance.item())
        param_history_lbfgs_v3["lengthscale_x"].append(model.kernel_X.lengthscale.item())
        param_history_lbfgs_v3["variance_x"].append(model.kernel_X.variance.item())

    if idx % 5 == 0:
        print(f"Iteration {idx}: \t Loss: {losses_lbfgs_v3[-1]}")

# Plot the results
plot_result(losses_lbfgs_v3, param_history_lbfgs_v3, true_params)