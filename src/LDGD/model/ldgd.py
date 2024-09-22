from ..model.utils.variational import VariationalLatentVariable, CholeskeyVariationalDist
from ..model.utils.variational import SharedVariationalStrategy
from ..visualization import plot_loss
from ..utils import dicts_to_dict_of_lists, check_one_hot_and_get_accuracy

import torch
import gpytorch
import numpy as np
import torch.nn as nn
from torch import optim
from .base import AbstractLDGD
from sklearn.decomposition import PCA


class LDGD(AbstractLDGD):
    def __init__(self, y, kernel_reg, kernel_cls, likelihood_reg, likelihood_cls,
                 latent_dim=2,
                 num_inducing_points_reg=10,
                 num_inducing_points_cls=10,
                 num_classes=3,
                 inducing_points=None,
                 use_gpytorch=True,
                 shared_inducing_points=False,
                 use_shared_kernel=False,
                 cls_weight=1.0,
                 reg_weight=1.0,
                 random_state=None,
                 device=None,
                 x_init=None):
        super().__init__(y, kernel_reg, kernel_cls, likelihood_reg, likelihood_cls,
                         latent_dim=latent_dim,
                         num_inducing_points_reg=num_inducing_points_reg,
                         num_inducing_points_cls=num_inducing_points_cls,
                         num_classes=num_classes,
                         inducing_points=inducing_points,
                         use_gpytorch=use_gpytorch,
                         shared_inducing_points=shared_inducing_points,
                         use_shared_kernel=use_shared_kernel,
                         cls_weight=cls_weight,
                         reg_weight=reg_weight,
                         random_state=random_state,
                         device=device)

        self.attempts = 1
        if x_init == 'pca':
            x_init = torch.Tensor(PCA(n_components=self.q).fit_transform(y))
        else:
            x_init = torch.randn(self.n, self.q)
        if self.use_gpytorch is True:
            x_prior_mean = torch.zeros(self.n, self.q, device=self.device)  # shape: N x Q
            self.prior_x = gpytorch.priors.NormalPrior(x_prior_mean, torch.ones_like(x_prior_mean, device=self.device))
            # self.x = VariationalLatentVariable(self.n, self.d, self.q, x_init, self.prior_x)
            self.x = gpytorch.models.gplvm.VariationalLatentVariable(self.n, self.d, self.q, x_init, self.prior_x)

            self.q_u_reg = gpytorch.variational.CholeskyVariationalDistribution(self.m_reg,
                                                                                batch_shape=self.batch_shape_reg)
            self.q_f_reg = gpytorch.variational.VariationalStrategy(model=self,
                                                                    inducing_points=self.inducing_inputs_reg,
                                                                    variational_distribution=self.q_u_reg,
                                                                    learn_inducing_locations=True)

            self.q_u_cls = gpytorch.variational.CholeskyVariationalDistribution(self.m_cls,
                                                                                batch_shape=self.batch_shape_cls)
            self.q_f_cls = gpytorch.variational.VariationalStrategy(model=self,
                                                                    inducing_points=self.inducing_inputs_cls,
                                                                    variational_distribution=self.q_u_cls,
                                                                    learn_inducing_locations=True)
        else:
            self.prior_x = torch.distributions.Normal(torch.zeros(self.n, self.q, device=self.device),
                                                      torch.ones(self.n, self.q, device=self.device))
            self.x = VariationalLatentVariable(self.n, self.d, self.q, X_init=x_init, prior_x=self.prior_x)

            self.q_u_reg = CholeskeyVariationalDist(num_inducing_points=self.m_reg, batch_shape=self.d)
            self.q_u_cls = CholeskeyVariationalDist(num_inducing_points=self.m_cls, batch_shape=self.k)

            self.q_f_cls = SharedVariationalStrategy(inducing_points=self.inducing_inputs_cls,
                                                     variational_distribution=self.q_u_cls,
                                                     jitter=self.jitter)

            self.q_f_reg = SharedVariationalStrategy(inducing_points=self.inducing_inputs_reg,
                                                     variational_distribution=self.q_u_reg,
                                                     jitter=self.jitter)

            self.inducing_inputs_cls = nn.Parameter(self.inducing_inputs_cls)
            self.inducing_inputs_reg = nn.Parameter(self.inducing_inputs_reg)

        self.to(device=self.device)

    def sample_latent_variable(self):
        return self.x()

    def forward(self, X, **kwargs):
        mode = kwargs.get('mode', 'reg')
        mean_x = self.mean_module(X)
        if mode == 'cls':
            covar_x = self.kernel_cls(X, X).add_jitter(self.jitter)
        else:
            covar_x = self.kernel_reg(X, X).add_jitter(self.jitter)
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return dist

    def train_model(self, yn, ys, learning_rate=0.01, epochs=100, batch_size=100, early_stop=None,
                    show_plot=False, monitor_mse=False, verbos=1, **kwargs):
        if kwargs.get('disp_interval') is not None:
            disp_interval = kwargs.get('disp_interval')
        else:
            disp_interval = 10


        if kwargs.get('save_best_result') is not None:
            save_best_result = kwargs.get('save_best_result')
        else:
            save_best_result = False

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses, x_mu_list, x_sigma_list, z_list_cls, z_list_reg = [], [], [], [], []
        loss_terms = []
        best_acc = 0.0
        best_loss = 999999
        for epoch in range(epochs):
            batch_index = self._get_batch_idx(batch_size, self.n)

            optimizer.zero_grad()
            x = self.sample_latent_variable()
            sample_batch = x[batch_index]

            # send files to GPU
            sample_batch = sample_batch.to(self.device)
            yn_batch = yn[batch_index].to(self.device)
            ys_batch = ys[batch_index].to(self.device)

            # Calculate loss
            try:
                loss, loss_dict = self.elbo(sample_batch, self.x, yn_batch, ys_batch)
                losses.append(loss.item())
                loss_terms.append(loss_dict)

                # Back propagate error
                loss.backward()
                optimizer.step()

                if epoch % disp_interval == 0:
                    mse_loss = self.update_history_train(yn=yn, elbo_loss=loss.item(), monitor_mse=monitor_mse)
                    loss_dict['mse_loss'] = mse_loss

                    predicted_ys_train, *_ = self.classify_x(self.x.q_mu)
                    accuracy_train = check_one_hot_and_get_accuracy(ys, predicted_ys_train)
                    loss_dict['accuracy'] = accuracy_train

                    if save_best_result is True:
                        if accuracy_train >= best_acc and loss.item()<best_loss:
                            best_acc = accuracy_train
                            best_loss = loss.item()
                            self.save_wights(path_save=kwargs.get('path_save'))
                            print("model saved!")

                    if verbos == 1:
                        if monitor_mse is True:
                            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, MSE: {mse_loss}, Accuracy: {accuracy_train}")
                        else:
                            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy_train}")
                    # print(f"f_reg {self.q_f_reg.jitter_val}, f_cls: {self.q_f_cls.jitter_val}")
                    if early_stop is not None:
                        if len(losses) > 100:
                            ce = np.abs(losses[-1] - np.mean(losses[-100:])) / np.abs(np.mean(losses[-100:]))
                            if ce < early_stop:
                                print("Early stop")
                                break
            except:  # Replace SomeSpecificException with the actual exception you expect
                if self.attempts < 0:
                    self.attempts += 1
                    self.jitter += 1e-4  # Increase jitter
                    print(
                        f"Attempt {self.attempts}: Error occurred, increasing jitter. New jitter: {self.jitter}")

        combined_dict = dicts_to_dict_of_lists(loss_terms)
        if show_plot is True:
            plot_loss(combined_dict)
            plot_loss(losses)

        return losses, combined_dict, self.history_train

    def predict_class(self, yn_test, ys_test, learning_rate=0.01, epochs=100, batch_size=100, early_stop=None, monitor_mse=False,
                      verbos=1):
        if yn_test.shape[1] != self.d:
            raise ValueError(f"yn_test should be of size [num_test, {self.d}]")

        for param in self.parameters():
            param.requires_grad = False

        self.history_test = {
            'elbo_loss': [],
            'x_mu_list': [],
            'x_sigma_list': [],
            'z_list_cls': [],
            'z_list_reg': [],
            'mse_loss': []
        }

        self.n_test = yn_test.shape[0]

        ys_test_list = []
        for k in range(self.k):
            y_test = torch.zeros(self.n_test, self.k, device=self.device)
            y_test[:, k] = 1
            ys_test_list.append(y_test)

        X_init = torch.randn(self.n_test, self.q)
        if self.use_gpytorch is True:
            x_prior_mean = torch.zeros(self.n_test, self.q, device=self.device)
            prior_x = gpytorch.priors.NormalPrior(x_prior_mean, torch.ones_like(x_prior_mean, device=self.device))
            self.x_test = gpytorch.models.gplvm.VariationalLatentVariable(self.n_test, self.d, self.q, X_init,
                                                                          prior_x).to(self.device)
        else:
            prior_x_test = torch.distributions.Normal(torch.zeros(self.n_test, self.q, device=self.device),
                                                      torch.ones(self.n_test, self.q, device=self.device))
            self.x_test = VariationalLatentVariable(self.n_test, self.d, self.q, X_init=X_init,
                                                    prior_x=prior_x_test).to(self.device)

        params_to_optimize = self.x_test.parameters()
        optimizer = optim.Adam(params_to_optimize, lr=learning_rate)


        losses, loss_terms, x_mu_list, x_sigma_list, z_list_reg, z_list_cls = [], [], [], [], [], []
        for epoch in range(epochs):
            batch_index = self._get_batch_idx(batch_size, self.n_test)
            optimizer.zero_grad()
            x_test = self.x_test()
            x_test_sampled = x_test[batch_index].to(self.device)
            yn_test_batch = yn_test[batch_index].to(self.device)
            try:
                loss, loss_dict = self.elbo(x_samples=x_test_sampled,
                                            x=self.x_test,
                                            yn=yn_test_batch)

                losses.append(loss.item())
                loss.backward(retain_graph=True)
                optimizer.step()

                if epoch % 10 == 0:
                    mse_loss = self.update_history_test(yn_test, elbo_loss=loss.item(), monitor_mse=monitor_mse)
                    loss_dict['mse_loss'] = mse_loss
                    predicted_ys_test, *_ = self.classify_x(self.x_test.q_mu)
                    accuracy_test = check_one_hot_and_get_accuracy(ys_test, predicted_ys_test)
                    loss_dict['accuracy'] = accuracy_test
                    if verbos == 1:
                        if monitor_mse is True:
                            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, MSE: {mse_loss}, Accuracy: {accuracy_test}")
                        else:
                            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy_test}")

                    loss_terms.append(loss_dict)

                    if early_stop is not None:
                        if len(losses) > 100:
                            ce = np.abs(losses[-1] - np.mean(losses[-100:])) / np.abs(np.mean(losses[-100:]))
                            if ce < early_stop:
                                print("Early stop")
                                break

            except:  # Replace SomeSpecificException with the actual exception you expect
                if self.attempts < 0:
                    self.attempts += 1
                    self.jitter += 1e-4  # Increase jitter
                    print(
                        f"Attempt {self.attempts}: Error occurred, increasing jitter. New jitter: {self.jitter}")


        predictions, *_ = self.classify_x(self.x_test.q_mu)
        combined_dict = dicts_to_dict_of_lists(loss_terms)
        return predictions, self.history_test, combined_dict

    def update_history_train(self, yn, elbo_loss, monitor_mse=False):

        if self.use_gpytorch is False:
            x_mu = self.x.q_mu.cpu().detach().numpy()
            x_sigma = self.x.q_sigma.cpu().detach().numpy()
            # x_mu = self.x.q_mu.cpu().detach().numpy()
            # x_sigma = torch.nn.functional.softplus(self.x.q_log_sigma.cpu()).detach().numpy()
            self.history_train['elbo_loss'].append(elbo_loss)
            self.history_train['x_mu_list'].append(x_mu.copy())
            self.history_train['x_sigma_list'].append(x_sigma.copy())
            self.history_train['z_list_cls'].append(self.inducing_inputs_cls.cpu().detach().numpy().copy())
            self.history_train['z_list_reg'].append(self.inducing_inputs_reg.cpu().detach().numpy().copy())
        else:
            x_mu = self.x.q_mu.cpu().detach().numpy()
            x_sigma = torch.nn.functional.softplus(self.x.q_log_sigma.cpu()).detach().numpy()
            self.history_train['x_mu_list'].append(x_mu.copy())
            self.history_train['x_sigma_list'].append(x_sigma.copy())
            self.history_train['z_list_cls'].append(self.q_f_cls.inducing_points.cpu().detach().numpy().copy())
            self.history_train['z_list_reg'].append(self.q_f_reg.inducing_points.cpu().detach().numpy().copy())

        if monitor_mse is True:
            predicted_yn, predicted_yn_std = self.regress_x(x_mu)
            mse_loss = np.mean(np.square(yn.cpu().detach().numpy() - predicted_yn.cpu().detach().numpy()))
            self.history_train['mse_loss'].append(mse_loss)
        else:
            mse_loss = None
        return mse_loss

    def update_history_test(self, yn_test, elbo_loss, monitor_mse=False):
        if self.use_gpytorch is False:
            x_mu = self.x_test.q_mu.cpu().detach().numpy()
            x_sigma = self.x_test.q_sigma.cpu().detach().numpy()
            self.history_test['elbo_loss'].append(elbo_loss)
            self.history_test['x_mu_list'].append(x_mu.copy())
            self.history_test['x_sigma_list'].append(x_sigma.copy())
            self.history_test['z_list_cls'].append(self.inducing_inputs_cls.cpu().detach().numpy().copy())
            self.history_test['z_list_reg'].append(self.inducing_inputs_reg.cpu().detach().numpy().copy())
        else:
            x_mu = self.x_test.q_mu.cpu().detach().numpy()
            x_sigma = self.x_test.q_log_sigma.cpu().detach().numpy()
            self.history_test['x_mu_list'].append(x_mu.copy())
            self.history_test['x_sigma_list'].append(x_sigma.copy())
            self.history_test['z_list_cls'].append(self.q_f_cls.inducing_points.cpu().detach().numpy().copy())
            self.history_test['z_list_reg'].append(self.q_f_reg.inducing_points.cpu().detach().numpy().copy())
        if monitor_mse is True:
            predicted_yn, predicted_yn_std = self.regress_x(x_mu)
            mse_loss = np.mean(np.square(yn_test.cpu().detach().numpy() - predicted_yn.cpu().detach().numpy()))
            self.history_test['mse_loss'].append(mse_loss)
        else:
            mse_loss = None
        return mse_loss
