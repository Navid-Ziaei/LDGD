import matplotlib.pyplot as plt
import torch
from src import Settings, Paths
from src import *
from src import ARDRBFKernel
from gp_project_pytorch.data.data_loader import load_dataset
from src import GPLVM_point_estimate, bGPLVM
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.io import savemat, loadmat

from gp_project_pytorch.visualization.animated_visualization import animate_train
from gp_project_pytorch.visualization.vizualize_utils import plot_heatmap

# Set the seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# load settings from settings.json
settings = Settings()
settings.load_settings()

# load device paths from device.json and create result paths
paths = Paths(settings)
paths.load_device_paths()

model_settings = {
    'latent_dim': 2,
    'num_inducing_points': 5,
    'num_epochs_train': 5000,
    'num_epochs_test': 5000,
    'batch_size': 200,
    'load_trained_model': False,
    'load_tested_model': False,
    'use_gpytorch': True,
    'n_features': 10,
    'dataset': settings.dataset
}
latent_dim = 2

# load raw data

"""yn_train, yn_test, ys_train, ys_test, labels_train, labels_test = load_dataset(dataset_name='iris',
                                                                               test_size=0.2,
                                                                               n_features=model_settings['n_features'])
train_data = {
    'yn_train': yn_train.detach().numpy(),
    'ys_train': ys_train.detach().numpy(),
    'labels_train': labels_train.detach().numpy()
}

test_data = {
    'yn_test': yn_test.detach().numpy(),
    'ys_test': ys_test.detach().numpy(),
    'labels_test': labels_test.detach().numpy()
}

savemat('D:\\Navid\Projects\\gp_project_pytorch\\data\\train_data_iris.mat', train_data)
savemat('D:\\Navid\Projects\\gp_project_pytorch\\data\\test_data_iris.mat', test_data)

np.save('D:\\Navid\Projects\\gp_project_pytorch\\data\\train_data_iris.npy', train_data)
np.save('D:\\Navid\Projects\\gp_project_pytorch\\data\\test_data_iris.npy', test_data)"""

train_data = np.load('/data/train_data.npy', allow_pickle=True)
test_data = np.load('/data/test_data.npy', allow_pickle=True)
yn_train, ys_train, labels_train = train_data.take(0)['yn_train'], train_data.take(0)['ys_train'], train_data.take(0)[
    'labels_train']
yn_test, ys_test, labels_test = test_data.take(0)['yn_test'], test_data.take(0)['ys_test'], test_data.take(0)[
    'labels_test']

yn_train, ys_train, labels_train = torch.Tensor(yn_train), torch.Tensor(ys_train), torch.Tensor(labels_train)
yn_test, ys_test, labels_test = torch.Tensor(yn_test), torch.Tensor(ys_test), torch.Tensor(labels_test)
N = yn_train.shape[0]
data_dim = yn_train.shape[1]
latent_dim = 2
n_inducing = 25
pca = False
color_list = ['r', 'g', 'b']
display_individuals = False
colors_train = [color_list[int(label)] for label in list(labels_train.detach())]
"==================================== PCA ======================================"
X_pca = PCA(n_components=2).fit_transform(yn_train)
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_train)
    plt.show()

"==================================== T-SNE ======================================"
try:
    x_tsne = np.load('/data/x_tsne_oil.npy')
except:
    x_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(yn_train)
    np.save('/data/x_tsne_oil.npy', x_tsne)
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors_train)
    plt.show()

"==================================== Point Estimate ======================================"
try:
    x_param_gplvm = np.load('/data/x_param_gplvm_oil.npy')
except:
    model = bGPLVM(N, data_dim, latent_dim, n_inducing, pca=pca, mode='point_estimate')

    # Likelihood
    likelihood = GaussianLikelihood(batch_shape=model.batch_shape)
    mll = VariationalELBO(likelihood, model, num_data=len(yn_train))

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}], lr=0.01)

    loss_list = model.train_model(yn_train, optimizer, mll, epochs=3000)

    x_param_gplvm = model.X.X.detach().numpy()
    np.save('/data/x_param_gplvm_oil.npy', x_param_gplvm)
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_param_gplvm[:, 0], x_param_gplvm[:, 1], c=colors_train)
    plt.show()
"==================================== Bayesian Estimate ======================================"
try:
    x_bayesian_gplvm = np.load('/data/x_bayesian_gplvm_oil.npy')
except:
    model = bGPLVM(N, data_dim, latent_dim, n_inducing, pca=pca, mode='bayesian')

    # Likelihood
    likelihood = GaussianLikelihood(batch_shape=model.batch_shape)
    mll = VariationalELBO(likelihood, model, num_data=len(yn_train))

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}], lr=0.01)

    loss_list = model.train_model(yn_train, optimizer, mll, epochs=5000)

    # vISUALIZATION
    x_bayesian_gplvm = model.X.q_mu.detach().numpy()
    np.save('/data/x_bayesian_gplvm_oil.npy', x_bayesian_gplvm)
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_bayesian_gplvm[:, 0], x_param_gplvm[:, 1], c=colors_train)
    plt.show()

"==================================== DBGPLVM Estimate ======================================"
try:
    x_dbgplvm = np.load('/data/x_dbgplvm_oil.npy')
except:
    model_settings['data_dim'] = yn_train.shape[-1]
    batch_shape = torch.Size([model_settings['data_dim']])
    model_settings['use_gpytorch'] = True
    if model_settings['use_gpytorch'] is False:
        kernel_cls = ARDRBFKernel(input_dim=model_settings['latent_dim'])
        kernel_reg = ARDRBFKernel(input_dim=model_settings['latent_dim'])
    else:
        kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))
        kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))

    likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
    likelihood_cls = BernoulliLikelihood()

    model = JointGPLVM_Bayesian(yn_train,
                                kernel_reg=kernel_reg,
                                kernel_cls=kernel_cls,
                                num_classes=ys_train.shape[-1],
                                latent_dim=model_settings['latent_dim'],
                                num_inducing_points=10,
                                likelihood_reg=likelihood_reg,
                                likelihood_cls=likelihood_cls,
                                use_gpytorch=model_settings['use_gpytorch'],
                                shared_inducing_points=True,
                                use_shared_kernel=False,
                                cls_weight=1,
                                reg_weight=1)

    losses, x_mu_list, x_sigma_list = model.train_model(yn=yn_train,
                                                        ys=ys_train,
                                                        epochs=7000,
                                                        batch_size=model_settings['batch_size'])
    model.save_wights(path_save=paths.path_model[0])

    with open(paths.path_model[0] + 'model_settings.json', 'w') as f:
        json.dump(model_settings, f, indent=2)

    x_dbgplvm = model.x.q_mu.detach().numpy()
    plt.figure(figsize=(10, 10))
    plt.scatter(x_dbgplvm[:, 0], x_dbgplvm[:, 1], c=colors_train)
    plt.show()

    np.save('/data/x_dbgplvm_oil.npy', x_dbgplvm)
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_bayesian_gplvm[:, 0], x_bayesian_gplvm[:, 1], c=colors_train)
    plt.show()

"==================================== ُُSPLVM Estimate ======================================"

results_sgplvm = loadmat("/data/result_sgplvm_oil.mat")
x_sgplvm = results_sgplvm['latent_z']
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_sgplvm[:, 0], x_sgplvm[:, 1], c=colors_train)
    plt.show()

"==================================== ُُSLLGPLVM Estimate ======================================"

results_sllgplvm = loadmat("/data/result_sllgplvm.mat")
x_sllgplvm = results_sllgplvm['zz']
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_sllgplvm[:, 0], x_sllgplvm[:, 1], c=colors_train)
    plt.show()

"==================================== fgplvm Estimate ======================================"

results_fgplvm = loadmat("/data/result_fgplvm.mat")
x_fgplvm = results_fgplvm['z']
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_fgplvm[:, 0], x_fgplvm[:, 1], c=colors_train)
    plt.show()

"==================================== Comparison ======================================"

fig, ax = plt.subplots(2, 8, figsize=(50, 16))
fontsize = 28  # Variable for consistent font size

color_list = ['r', 'b', 'g']
label_list = ['class 1', 'class 2', 'class 3']  # Assuming there's a typo in your original label list

# Plotting data
for i in range(3):
    ax[0, 0].scatter(X_pca[labels_train == i, 0], X_pca[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[0, 1].scatter(x_tsne[labels_train == i, 0], x_tsne[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[0, 2].scatter(x_param_gplvm[labels_train == i, 0], x_param_gplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[0, 3].scatter(x_bayesian_gplvm[labels_train == i, 0], x_bayesian_gplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[0, 4].scatter(x_fgplvm[labels_train == i, 0], x_fgplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[0, 5].scatter(x_sgplvm[labels_train == i, 0], x_sgplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[0, 5].set_xlim([np.quantile(x_sgplvm[labels_train == i, 0], 0.02),
                       np.quantile(x_sgplvm[labels_train == i, 0], 0.98)])
    ax[0, 5].set_ylim([np.quantile(x_sgplvm[labels_train == i, 1], 0.02),
                       np.quantile(x_sgplvm[labels_train == i, 1], 0.98)])
    ax[0, 6].scatter(x_sllgplvm[labels_train == i, 0], x_sllgplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[0, 7].scatter(x_dbgplvm[labels_train == i, 0], x_dbgplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])

for i in range(8):
    ax[0, i].set_xlabel('$X_1$', fontsize=fontsize)
    ax[0, i].set_ylabel('$X_2$', fontsize=fontsize)
    ax[0, i].tick_params(axis='both', which='major', labelsize=fontsize)

    ax[0, i].spines['top'].set_visible(False)
    ax[0, i].spines['right'].set_visible(False)

# plt.tight_layout()
# plt.show()
# fig.savefig("D:\\Navid\\Projects\\gp_project_pytorch\\data\\oil_dataset_comparison.png")
# fig.savefig("D:\\Navid\\Projects\\gp_project_pytorch\\data\\oil_dataset_comparison.svg")


# =================================== Iris=======================================

train_data = np.load('/data/train_data_iris.npy', allow_pickle=True)
test_data = np.load('/data/test_data_iris.npy', allow_pickle=True)
yn_train, ys_train, labels_train = train_data.take(0)['yn_train'], train_data.take(0)['ys_train'], train_data.take(0)[
    'labels_train']
yn_test, ys_test, labels_test = test_data.take(0)['yn_test'], test_data.take(0)['ys_test'], test_data.take(0)[
    'labels_test']

yn_train, ys_train, labels_train = torch.Tensor(yn_train), torch.Tensor(ys_train), torch.Tensor(labels_train)
yn_test, ys_test, labels_test = torch.Tensor(yn_test), torch.Tensor(ys_test), torch.Tensor(labels_test)
N = yn_train.shape[0]
data_dim = yn_train.shape[1]
latent_dim = 2
n_inducing = 25
pca = False
color_list = ['r', 'g', 'b']
colors_train = [color_list[int(label)] for label in list(labels_train.detach())]
"==================================== PCA ======================================"
X_pca = PCA(n_components=2).fit_transform(yn_train)
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors_train)
    plt.show()

"==================================== T-SNE ======================================"
try:
    x_tsne = np.load('/data/x_tsne_iris.npy')
except:
    x_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(yn_train)
    np.save('/data/x_tsne_iris.npy', x_tsne)
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=colors_train)
    plt.show()

"==================================== Point Estimate ======================================"
try:
    x_param_gplvm = np.load('/data/x_param_gplvm_iris.npy')
except:
    model = bGPLVM(N, data_dim, latent_dim, n_inducing, pca=pca, mode='point_estimate')

    # Likelihood
    likelihood = GaussianLikelihood(batch_shape=model.batch_shape)
    mll = VariationalELBO(likelihood, model, num_data=len(yn_train))

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}], lr=0.01)

    loss_list = model.train_model(yn_train, optimizer, mll, epochs=3000)

    x_param_gplvm = model.X.X.detach().numpy()
    np.save('/data/x_param_gplvm_iris.npy', x_param_gplvm)
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_param_gplvm[:, 0], x_param_gplvm[:, 1], c=colors_train)
    plt.show()
"==================================== Bayesian Estimate ======================================"
try:
    x_bayesian_gplvm = np.load('/data/x_bayesian_gplvm_iris.npy')
except:
    model = bGPLVM(N, data_dim, latent_dim, n_inducing, pca=pca, mode='bayesian')

    # Likelihood
    likelihood = GaussianLikelihood(batch_shape=model.batch_shape)
    mll = VariationalELBO(likelihood, model, num_data=len(yn_train))

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}], lr=0.01)

    loss_list = model.train_model(yn_train, optimizer, mll, epochs=3000)

    # vISUALIZATION
    x_bayesian_gplvm = model.X.q_mu.detach().numpy()
    np.save('/data/x_bayesian_gplvm_iris.npy', x_bayesian_gplvm)
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_bayesian_gplvm[:, 0], x_param_gplvm[:, 1], c=colors_train)
    plt.show()

"==================================== DBGPLVM Estimate ======================================"
try:
    x_dbgplvm = np.load('/data/x_dbgplvm_iris.npy')
except:
    model_settings['data_dim'] = yn_train.shape[-1]
    batch_shape = torch.Size([model_settings['data_dim']])
    model_settings['use_gpytorch'] = True
    if model_settings['use_gpytorch'] is False:
        kernel_cls = ARDRBFKernel(input_dim=model_settings['latent_dim'])
        kernel_reg = ARDRBFKernel(input_dim=model_settings['latent_dim'])
    else:
        kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))
        kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=model_settings['latent_dim']))

    likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
    likelihood_cls = BernoulliLikelihood()

    model = JointGPLVM_Bayesian(yn_train,
                                kernel_reg=kernel_reg,
                                kernel_cls=kernel_cls,
                                num_classes=ys_train.shape[-1],
                                latent_dim=model_settings['latent_dim'],
                                num_inducing_points=5,
                                likelihood_reg=likelihood_reg,
                                likelihood_cls=likelihood_cls,
                                use_gpytorch=model_settings['use_gpytorch'],
                                shared_inducing_points=True,
                                use_shared_kernel=False,
                                cls_weight=1,
                                reg_weight=1)

    losses, x_mu_list, x_sigma_list = model.train_model(yn=yn_train,
                                                        ys=ys_train,
                                                        epochs=7000,
                                                        batch_size=model_settings['batch_size'])
    model.save_wights(path_save=paths.path_model[0])

    with open(paths.path_model[0] + 'model_settings.json', 'w') as f:
        json.dump(model_settings, f, indent=2)
    x_dbgplvm = model.x.q_mu.detach().numpy()
    np.save('/data/x_dbgplvm_iris.npy', x_dbgplvm)
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_dbgplvm[:, 0], x_dbgplvm[:, 1], c=colors_train)
    plt.show()

"==================================== ُُSGPLVM Estimate ======================================"

results_sgplvm = loadmat("/data/result_sgplvm_iris.mat")
x_sgplvm = results_sgplvm['latent_z']
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_sgplvm[:, 0], x_sgplvm[:, 1], c=colors_train)
    plt.show()

"==================================== ُُSLLGPLVM Estimate ======================================"

results_sllgplvm = loadmat("/data/result_sllgplvm_iris.mat")
x_sllgplvm = results_sllgplvm['z']
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_sllgplvm[:, 0], x_sllgplvm[:, 1], c=colors_train)
    plt.show()

"==================================== fgplvm Estimate ======================================"

results_fgplvm = loadmat("/data/result_fgplvm_iris.mat")
x_fgplvm = results_fgplvm['z']
if display_individuals is True:
    plt.figure(figsize=(10, 10))
    plt.scatter(x_fgplvm[:, 0], x_fgplvm[:, 1], c=colors_train)
    plt.show()

"==================================== Comparison ======================================"

# fig, ax = plt.subplots(2, 8, figsize=(50, 16))
# fontsize = 28  # Variable for consistent font size

# color_list = ['r', 'b', 'g']
# label_list = ['class 1', 'class 2', 'class 3']  # Assuming there's a typo in your original label list

# Plotting data
for i in range(3):
    ax[1, 0].scatter(X_pca[labels_train == i, 0], X_pca[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])

    ax[1, 1].scatter(x_tsne[labels_train == i, 0], x_tsne[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[1, 2].scatter(x_param_gplvm[labels_train == i, 0], x_param_gplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[1, 3].scatter(x_bayesian_gplvm[labels_train == i, 0], x_bayesian_gplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[1, 4].scatter(x_fgplvm[labels_train == i, 0], x_fgplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[1, 5].scatter(x_sgplvm[labels_train == i, 0], x_sgplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[1, 5].set_xlim([np.quantile(x_sgplvm[labels_train == i, 0], 0.02),
                       np.quantile(x_sgplvm[labels_train == i, 0], 0.98)])
    ax[1, 5].set_ylim([np.quantile(x_sgplvm[labels_train == i, 1], 0.02),
                       np.quantile(x_sgplvm[labels_train == i, 1], 0.98)])
    ax[1, 6].scatter(x_sllgplvm[labels_train == i, 0], x_sllgplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
    ax[1, 7].scatter(x_dbgplvm[labels_train == i, 0], x_dbgplvm[labels_train == i, 1], c=color_list[i],
                     s=40, alpha=1, edgecolor=color_list[i])
titles_list = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
for i in range(8):
    ax[0, i].set_title(titles_list[i], fontsize=34)
    ax[1, i].set_xlabel('$X_1$', fontsize=fontsize)
    ax[1, i].set_ylabel('$X_2$', fontsize=fontsize)
    ax[1, i].tick_params(axis='both', which='major', labelsize=fontsize)

    ax[1, i].spines['top'].set_visible(False)
    ax[1, i].spines['right'].set_visible(False)

plt.tight_layout()
plt.show()
fig.savefig("D:\\Navid\\Projects\\gp_project_pytorch\\data\\iris_dataset_comparison.png")
fig.savefig("D:\\Navid\\Projects\\gp_project_pytorch\\data\\iris_dataset_comparison.svg")
