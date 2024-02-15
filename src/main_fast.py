import json

import gpytorch
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.LDGD.visualization import plot_results_gplvm
from src.LDGD.settings import Settings, Paths
from src.LDGD.model.utils import ARDRBFKernel
from src.LDGD.model import FastLDGD, LDGD
from src.LDGD.visualization.animated_visualization import animate_train
from src.LDGD.visualization.vizualize_utils import plot_heatmap
from src.LDGD.data.data_loader import load_dataset
from src.LDGD.model.variational_autoencoder import VAE

from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood

# Set the seed for reproducibility
random_state = None
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load settings from settings.json
settings = Settings()
settings.load_settings()

# load device paths from device.json and create result paths
paths = Paths(settings)
paths.load_device_paths()

model_settings = {
    'latent_dim': 5,
    'num_inducing_points_reg': 100,
    'num_inducing_points_cls': 30,
    'num_inducing_points': 5,
    'num_epochs_train': 1500,
    'num_epochs_test': 1500,
    'batch_size': 100,
    'load_trained_model': False,
    'load_tested_model': False,
    'use_gpytorch': True,
    'n_features': 10,
    'dataset': settings.dataset,
    'use_shared_kernel': False,
    'shared_inducing_points': False,
    'reg_weight': 1.0,
    'cls_weight': 1.0
}

# load raw data
yn_train, yn_test, ys_train, ys_test, labels_train, labels_test = load_dataset(dataset_name=settings.dataset,
                                                                               test_size=0.8,
                                                                               n_features=model_settings['n_features'],
                                                                               random_tate=random_state)
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(input_dim=yn_train.shape[-1],
            hidden_dim=50,
            latent_dim=2,
            num_classes=len(np.unique(labels_train))).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.fit(x=yn_train, y=labels_train, x_test=yn_test, y_test=labels_test, optimizer=optimizer, epochs=1000, batch_size=500)
y_hat, mean, log_var, metrics = model.evaluate(yn_test, labels_test)

plt.figure(figsize=(10, 10))
plt.scatter(mean[:, 0], mean[:, 1], c=labels_test, cmap='rainbow')
plt.show()
"""

model_settings['data_dim'] = yn_train.shape[-1]
batch_shape = torch.Size([model_settings['data_dim']])

if model_settings['use_gpytorch'] is False:
    kernel_cls = ARDRBFKernel(input_dim=model_settings['latent_dim'])
    kernel_reg = ARDRBFKernel(input_dim=model_settings['latent_dim'])
    kernel_reg = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim'])).to(device)
    kernel_cls = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim'])).to(
        device)
else:
    kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim'])).to(
        device)
    kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim'])).to(
        device)

likelihood_reg = GaussianLikelihood(batch_shape=batch_shape).to(device)
likelihood_cls = BernoulliLikelihood().to(device)

model = FastLDGD(yn_train,
                 kernel_reg=kernel_reg,
                 kernel_cls=kernel_cls,
                 num_classes=ys_train.shape[-1],
                 latent_dim=model_settings['latent_dim'],
                 num_inducing_points_reg=model_settings['num_inducing_points_reg'],
                 num_inducing_points_cls=model_settings['num_inducing_points_cls'],
                 num_inducing_points=model_settings['num_inducing_points'],
                 likelihood_reg=likelihood_reg,
                 likelihood_cls=likelihood_cls,
                 use_gpytorch=model_settings['use_gpytorch'],
                 shared_inducing_points=model_settings['shared_inducing_points'],
                 use_shared_kernel=model_settings['use_shared_kernel'],
                 cls_weight=model_settings['cls_weight'],
                 reg_weight=model_settings['reg_weight'],
                 random_state=random_state)

if model_settings['load_trained_model'] is False:
    losses, history_train = model.train_model(yn=yn_train, ys=ys_train,
                                              epochs=model_settings['num_epochs_train'],
                                              batch_size=model_settings['batch_size'], show_plot=True)
    model.save_wights(path_save=paths.path_model[0])

    with open(paths.path_model[0] + 'model_settings.json', 'w') as f:
        json.dump(model_settings, f, indent=2)
else:
    losses = []
    model.load_weights(paths.model)

if model.use_gpytorch_kernel is False:
    alpha_reg = model.kernel_reg.alpha.cpu().detach().numpy()
    alpha_cls = model.kernel_cls.alpha.cpu().detach().numpy()
    X = model.x.q_mu.cpu().detach().numpy()
    std = model.x.q_sigma.cpu().detach().numpy()
else:
    alpha_reg = 1 / model.kernel_reg.base_kernel.lengthscale.cpu().detach().numpy()
    alpha_cls = 1 / model.kernel_cls.base_kernel.lengthscale.cpu().detach().numpy()
    X, q_log_sigma = model.x.encode(yn_train.to(device))
    X = X.cpu().detach().numpy()
    std = torch.nn.functional.softplus(q_log_sigma).cpu().detach().numpy()

animate_train(history_train['x_mu_list'], labels_train, 'train_animation_with_inducing',
              save_path=paths.path_result[0],
              inverse_length_scale=alpha_cls,
              inducing_points_history=(history_train['z_list_reg'], history_train['z_list_cls']))

"""predicted_yn, predicted_yn_std = model.regress_x(np.array([[2, -1]]))
print(np.mean(np.mean(np.square(yn_train[labels_train == 0] - predicted_yn).detach().numpy(), 1)))
print(np.mean(np.mean(np.square(yn_train[labels_train == 1] - predicted_yn).detach().numpy(), 1)))
"""
predictions, metrics, history_test = model.evaluate(yn_test=yn_test, ys_test=labels_test,
                                                    epochs=model_settings['num_epochs_test'],
                                                    save_path=paths.path_result[0])

inducing_points = (history_train['z_list_reg'][-1], history_train['z_list_cls'][-1])
if model.use_gpytorch_kernel is False:
    alpha_reg = model.kernel_reg.alpha.cpu().detach().numpy()
    alpha_cls = model.kernel_cls.alpha.cpu().detach().numpy()
    x_test = model.x_test.q_mu.cpu().detach().numpy()
    std_test = model.x_test.q_sigma.cpu().detach().numpy()
else:
    x_test, q_log_sigma = model.x.encode(yn_test.to(device))
    x_test = x_test.cpu().detach().numpy()
    std_test = torch.nn.functional.softplus(q_log_sigma).cpu().detach().numpy()

plot_heatmap(X, labels_train, model, alpha_cls, cmap='winter', range_scale=1.2,
             file_name='latent_heatmap_train', inducing_points=inducing_points, save_path=paths.path_result[0])
plot_heatmap(x_test, labels_test, model, alpha_cls, cmap='winter', range_scale=1.2,
             file_name='latent_heatmap_test', inducing_points=inducing_points, save_path=paths.path_result[0])

plot_results_gplvm(X, std, labels=labels_train, losses=losses, inverse_length_scale=alpha_reg,
                   latent_dim=model_settings['latent_dim'],
                   save_path=paths.path_result[0], file_name='gplvm_train_reg_result')
plot_results_gplvm(X, std, labels=labels_train, losses=losses, inverse_length_scale=alpha_cls,
                   latent_dim=model_settings['latent_dim'],
                   save_path=paths.path_result[0], file_name='gplvm_train_cls_result')

plot_results_gplvm(x_test, std_test, labels=labels_test, losses=losses, inverse_length_scale=alpha_cls,
                   latent_dim=model_settings['latent_dim'],
                   save_path=paths.path_result[0], file_name='gplvm_test_result')

plt.show()

if model_settings['load_trained_model'] is False:
    animate_train(history_train['x_mu_list'], labels_train, 'train_animation', save_path=paths.path_result[0],
                  inverse_length_scale=alpha_cls)

