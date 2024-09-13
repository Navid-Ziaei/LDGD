import json

import gpytorch
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.LDGD.visualization import plot_results_gplvm
from src.LDGD.settings import Settings, Paths
from src.LDGD.model.utils import ARDRBFKernel
from src.LDGD.model import LDGD, JointGPLVM_Bayesian
from src.LDGD.visualization.animated_visualization import animate_train
from src.LDGD.visualization.vizualize_utils import plot_heatmap
from src.LDGD.data.data_loader import load_dataset
from src.LDGD.model.variational_autoencoder import VAE

from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
import winsound

duration = 800  # milliseconds
freq = 440  # Hz
# Set the seed for reproducibility
random_state = 42
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
    'latent_dim': 2,
    'num_inducing_points_reg': 5,
    'num_inducing_points_cls': 5,
    'num_epochs_train': 1500,
    'num_epochs_test': 1500,
    'batch_size': 100,
    'load_trained_model': False,
    'load_tested_model': False,
    'use_gpytorch': True,
    'n_features': 20,
    'dataset': settings.dataset,
    'shared_inducing_points': True,
    'reg_weight': 1.0,
    'cls_weight': 1.0
}

# load raw data
yn_train, yn_test, ys_train, ys_test, labels_train, labels_test = load_dataset(dataset_name=settings.dataset,
                                                                               test_size=0.2,
                                                                               n_features=model_settings['n_features'],
                                                                               random_tate=random_state,
                                                                               n_samples=300)

"""from LDGD.visualization import plot_2d_scatter
plot_2d_scatter(X=yn_train, y=labels_train, idx1=39, idx2=13)
plt.show()"""
# train_size = 10
# yn_train, ys_train, labels_train = yn_train[:train_size], ys_train[:train_size], labels_train[:train_size]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_settings['data_dim'] = yn_train.shape[-1]
batch_shape = torch.Size([model_settings['data_dim']])

if model_settings['use_gpytorch'] is False:
    kernel_cls = ARDRBFKernel(input_dim=model_settings['latent_dim'])
    kernel_reg = ARDRBFKernel(input_dim=model_settings['latent_dim'])
    """kernel_reg = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim'])).to(device)
    kernel_cls = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim'])).to(device)"""
else:
    kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
        ard_num_dims=model_settings['latent_dim'])).to(device)
    kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(
        ard_num_dims=model_settings['latent_dim'])).to(device)

likelihood_reg = GaussianLikelihood(batch_shape=batch_shape).to(device)
likelihood_cls = BernoulliLikelihood().to(device)

model = LDGD(yn_train,
             kernel_reg=kernel_reg,
             kernel_cls=kernel_cls,
             num_classes=ys_train.shape[-1],
             latent_dim=model_settings['latent_dim'],
             num_inducing_points_reg=model_settings['num_inducing_points_reg'],
             num_inducing_points_cls=model_settings['num_inducing_points_cls'],
             likelihood_reg=likelihood_reg,
             likelihood_cls=likelihood_cls,
             use_gpytorch=model_settings['use_gpytorch'],
             shared_inducing_points=model_settings['shared_inducing_points'],
             cls_weight=model_settings['cls_weight'],
             reg_weight=model_settings['reg_weight'],
             random_state=random_state,
             x_init=None,
             device=device)

if model_settings['load_trained_model'] is False:
    losses, combined_dict, history_train = model.train_model(yn=yn_train, ys=ys_train,
                                                             epochs=model_settings['num_epochs_train'],
                                                             batch_size=model_settings['batch_size'],
                                                             show_plot=True, early_stop=1e-6)
    model.save_wights(path_save=paths.path_model[0])

    with open(paths.path_model[0] + 'model_settings.json', 'w') as f:
        json.dump(model_settings, f, indent=2)
else:
    losses = []
    model.load_weights(paths.path_model[0])

if model.use_gpytorch_kernel is False:
    alpha_reg = model.kernel_reg.alpha.cpu().detach().numpy()
    alpha_cls = model.kernel_cls.alpha.cpu().detach().numpy()
    X = model.x.q_mu.cpu().detach().numpy()
    std = model.x.q_sigma.cpu().detach().numpy()
else:
    alpha_reg = 1 / model.kernel_reg.base_kernel.lengthscale.cpu().detach().numpy()
    alpha_cls = 1 / model.kernel_cls.base_kernel.lengthscale.cpu().detach().numpy()
    X = model.x.q_mu.cpu().detach().numpy()
    std = torch.nn.functional.softplus(model.x.q_log_sigma).cpu().detach().numpy()

label_pred, out_prob, out_std = model.classify_x(model.x.q_mu)

var_array = np.zeros_like(label_pred, 'float32')
for idx, var in enumerate(out_std):
    selected_var = var[label_pred == idx]
    var_array[label_pred == idx] = selected_var

animate_train(history_train['x_mu_list'], labels_train, 'train_animation_with_inducing',
              save_path=paths.path_result[0],
              inverse_length_scale=alpha_cls,
              inducing_points_history=(history_train['z_list_reg'], history_train['z_list_cls']))

"""predicted_yn, predicted_yn_std = model.regress_x(np.array([[2, -1]]))
print(np.mean(np.mean(np.square(yn_train[labels_train == 0] - predicted_yn).detach().numpy(), 1)))
print(np.mean(np.mean(np.square(yn_train[labels_train == 1] - predicted_yn).detach().numpy(), 1)))
"""
predictions, metrics, history_test, *_ = model.evaluate(yn_test=yn_test, ys_test=labels_test,
                                                        epochs=model_settings['num_epochs_test'],
                                                        save_path=paths.path_result[0])

animate_train(history_test['x_mu_list'], labels_test, 'test_animation_with_inducing',
              save_path=paths.path_result[0],
              inverse_length_scale=alpha_cls,
              inducing_points_history=(history_test['z_list_reg'], history_test['z_list_cls']))

inducing_points = (history_test['z_list_reg'][-1], history_test['z_list_cls'][-1])
if model.use_gpytorch_kernel is False:
    alpha_reg = model.kernel_reg.alpha.cpu().detach().numpy()
    alpha_cls = model.kernel_cls.alpha.cpu().detach().numpy()
    x_test = model.x_test.q_mu.cpu().detach().numpy()
    std_test = model.x_test.q_sigma.cpu().detach().numpy()
else:
    alpha_reg = 1 / model.kernel_reg.base_kernel.lengthscale.cpu().detach().numpy()
    alpha_cls = 1 / model.kernel_cls.base_kernel.lengthscale.cpu().detach().numpy()
    x_test = model.x_test.q_mu.cpu().detach().numpy()
    std_test = torch.nn.functional.softplus(model.x_test.q_log_sigma).cpu().detach().numpy()
    std_test = torch.nn.functional.softplus(model.x_test.q_log_sigma).cpu().detach().numpy()

plot_heatmap(X, labels_train, model, alpha_cls, cmap='binary', range_scale=1.2,
             file_name='latent_heatmap_train', inducing_points=inducing_points, save_path=paths.path_result[0],
             device=device,
             heat_map_mode='std', show_legend=False)
plot_heatmap(x_test, labels_test, model, alpha_cls, cmap='binary', range_scale=1.2,
             file_name='latent_heatmap_test', inducing_points=inducing_points, save_path=paths.path_result[0],
             device=device,
             heat_map_mode='std', show_legend=False)

plot_results_gplvm(X, np.sqrt(std), labels=labels_train, losses=losses, inverse_length_scale=alpha_reg,
                   latent_dim=model_settings['latent_dim'],
                   save_path=paths.path_result[0], file_name='gplvm_train_reg_result', Z=inducing_points)
plot_results_gplvm(X, np.sqrt(std), labels=labels_train, losses=losses, inverse_length_scale=alpha_cls,
                   latent_dim=model_settings['latent_dim'],
                   save_path=paths.path_result[0], file_name='gplvm_train_cls_result', Z=inducing_points)

plot_results_gplvm(x_test, np.sqrt(std_test), labels=labels_test, losses=losses, inverse_length_scale=alpha_cls,
                   latent_dim=model_settings['latent_dim'],
                   save_path=paths.path_result[0], file_name='gplvm_test_result', Z=inducing_points)

plt.show()
winsound.Beep(freq, duration * 3)
if model_settings['load_trained_model'] is False:
    animate_train(history_train['x_mu_list'], labels_train, 'train_animation', save_path=paths.path_result[0],
                  inverse_length_scale=alpha_cls)
animate_train(history_test['x_mu_list'], labels_test, 'test_animation', save_path=paths.path_result[0],
              inverse_length_scale=alpha_cls)
