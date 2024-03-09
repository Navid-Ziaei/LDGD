import gpytorch
import numpy as np
import torch
import winsound
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from LDGD.model import LDGD, FastLDGD, VAE
from LDGD.visualization.vizualize_utils import plot_heatmap, plot_2d_scatter, plot_ARD_gplvm
from LDGD.visualization.vizualize_utils import plot_loss_gplvm, plot_scatter_gplvm, plot_box_plots
from LDGD.data.data_loader import generate_data

from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood

from LDGD.utils import dicts_to_dict_of_lists
import json
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

duration = 1000  # milliseconds
freq = 440  # Hz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_LDGD_model(data_cont, data_cat, ldgd_settings, batch_shape, x_init='pca'):
    kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ldgd_settings['latent_dim']))
    kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ldgd_settings['latent_dim']))

    likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
    likelihood_cls = BernoulliLikelihood()
    model = LDGD(data_cont,
                 kernel_reg=kernel_reg,
                 kernel_cls=kernel_cls,
                 num_classes=data_cat.shape[-1],
                 latent_dim=ldgd_settings['latent_dim'],
                 num_inducing_points_cls=ldgd_settings['num_inducing_points_cls'],
                 num_inducing_points_reg=ldgd_settings['num_inducing_points_reg'],
                 likelihood_reg=likelihood_reg,
                 likelihood_cls=likelihood_cls,
                 use_gpytorch=ldgd_settings['use_gpytorch'],
                 shared_inducing_points=ldgd_settings['shared_inducing_points'],
                 use_shared_kernel=False,
                 x_init=x_init,
                 device=device)

    return model


# Assuming X, y are numpy arrays
def cross_validate_model(n_dim, settings, n_splits=5, load_saved_result=False, save_model=True, **kwargs):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=model_settings['random_state'])

    performances = []

    for n_dim in n_dim:
        # Code to generate data for different dimensional synthetic datasets
        pattern = kwargs.get('pattern', 'moon')  # default pattern
        n_samples = kwargs.get('n_samples', 1500)
        noise = kwargs.get('noise', 0.1)
        increase_method = kwargs.get('increase_method', 'linear')

        X, y, orig_data = generate_data(pattern, n_samples, noise, n_dim, increase_method,
                                        random_state=model_settings['random_state'])
        # One-hot encode the labels
        y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
        y_one_hot[np.arange(y.shape[0]), np.uint(y)] = 1

        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.float32)
        y = torch.tensor(y_one_hot)


        p, r, f = [], [], []
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X, labels)):
            # "X[train_ix]" will have to be modified to extract the relevant synthetic rows
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            labels_train, labels_test = labels[train_idx], labels[test_idx]

            # Here you convert to PyTorch tensors, etc., and pass into your network
            # metrics = train_and_evaluate_model() assuming it returns a dict or other
            # performance data. Replace with the appropiate flow of your process.
            settings['data_dim'] = X_train.shape[-1]
            batch_shape = torch.Size([settings['data_dim']])
            if load_saved_result is False:
                model = create_LDGD_model(X_train, y_train, settings, batch_shape=batch_shape, x_init='pca')
                print(X_train.shape)
                print(y_train.shape)
                losses, history_train = model.train_model(yn=X_train, ys=y_train,
                                                          epochs=settings['num_epochs_train'],
                                                          batch_size=settings['batch_size'])
                if save_model is True:
                    model.save_wights(path_save='./saved_models/', file_name=f"model_synthetic_fold{fold_idx}_{n_dim}")
                predictions, metrics, history_test = model.evaluate(yn_test=X_test, ys_test=labels_test,
                                                                    epochs=settings['num_epochs_test'])

                winsound.Beep(freq, duration)
            else:
                model = create_LDGD_model(X_train, y_train, settings)
                model.load_weights(path_save='./saved_models/', file_name=f'model_synthetic_fold{fold_idx}_{n_dim}.pth')
                predictions, metrics, history_test = model.evaluate(yn_test=X_test, ys_test=labels_test,
                                                                    epochs=settings['num_epochs_test'])

            # Compute or accumulate measures (precision, recall, F1 score)
            p.append(metrics['precision'])
            r.append(metrics['recall'])
            f.append(metrics['f1_score'])

        # Form the list of dataframe rows for the round of n_dim rounds
        measures = (np.mean(p), np.std(p), np.mean(r), np.std(r), np.mean(f), np.std(f))
        round_result = {"dataset": f"synthetic {n_dim}", "precision": f"{measures[0]:.2f} ± {measures[1]:.2f}",
                        "recall": f"{measures[2]:.2f} ± {measures[3]:.2f}",
                        "f score": f"{measures[4]:.2f} ± {measures[5]:.2f}"}
        performances.append(round_result)

    # Creating DataFrame from performance data
    df_performances = pd.DataFrame(performances)
    print(df_performances)
    return df_performances


model_settings = {
    'latent_dim': 2,
    'num_inducing_points': 5,
    'num_inducing_points_reg': 8,
    'num_inducing_points_cls': 8,
    'num_epochs_train': 50,
    'num_epochs_test': 50,
    'batch_size': 100,
    'load_trained_model': False,
    'load_tested_model': False,
    'dataset': 'synthetic',
    'shared_inducing_points': True,
    'use_gpytorch': True,
    'random_state': 65,
    'test_size': 0.8,
    'cls_weight': 1.0,
    'reg_weight': 1.0,
    'num_samples': 500,

}
cross_validate_model(n_dim=[5, 10, 20], settings=model_settings, n_splits=5, load_saved_result=False, save_model=True)
