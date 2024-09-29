from ..model.utils.variational import VariationalDist, VariationalLatentVariable, \
    CholeskeyVariationalDist, SharedVariationalStrategy, VariationalLatentVariableNN
from ..visualization import plot_loss
from ..model.utils.variational_strategy import VariationalStrategy2
from ..utils import dicts_to_dict_of_lists, check_one_hot_and_get_accuracy
import torch
import gpytorch
import numpy as np
import torch.nn as nn
from torch import optim
from .base import AbstractLDGD


class FastLDGD(AbstractLDGD):
    def __init__(self, y, kernel_reg, kernel_cls, likelihood_reg, likelihood_cls, latent_dim=2,
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
                 nn_encoder=None):
        super().__init__(y, kernel_reg, kernel_cls, likelihood_reg, likelihood_cls, latent_dim=latent_dim,
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

        x_init = torch.randn(self.n, self.q)
        if self.use_gpytorch is True:
            x_prior_mean = torch.zeros(self.n, self.q, device=self.device)  # shape: N x Q
            self.prior_x = gpytorch.priors.NormalPrior(x_prior_mean, torch.ones_like(x_prior_mean, device=self.device))
            self.x = VariationalLatentVariableNN(self.n, self.d, self.q, x_init, self.prior_x,
                                                 nn_encoder=nn_encoder)

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
            self.x = VariationalLatentVariableNN(self.n, self.d, self.q, X_init=x_init, prior_x=self.prior_x,
                                                 nn_encoder=nn_encoder)

            self.q_u_reg = CholeskeyVariationalDist(num_inducing_points=self.m_reg, batch_shape=self.d)
            self.q_u_cls = CholeskeyVariationalDist(num_inducing_points=self.m_cls, batch_shape=self.k)

            self.q_f_cls = SharedVariationalStrategy(inducing_points=self.inducing_inputs_cls,
                                                     variational_distribution=self.q_u_cls,
                                                     jitter=self.jitter)

            self.q_f_reg = SharedVariationalStrategy(inducing_points=self.inducing_inputs_reg,
                                                     variational_distribution=self.q_u_reg,
                                                     jitter=self.jitter)

            self.log_noise_sigma = nn.Parameter(torch.ones(self.d, device=self.device) * -2)

        self.to(device=self.device)

    def sample_latent_variable(self, yn):
        return self.x(yn)

    def forward(self, X, **kwargs):
        mode = kwargs.get('mode', 'reg')
        mean_x = self.mean_module(X)
        if mode == 'cls':
            covar_x = self.kernel_cls(X, X).add_jitter(self.jitter)
        else:
            covar_x = self.kernel_reg(X, X).add_jitter(self.jitter)
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return dist

    def train_model(self, yn, ys, learning_rate=0.01, epochs=100, batch_size=100, early_stop=None, show_plot=False,
                    monitor_mse=False,
                    yn_test=None, ys_test=None, **kwargs):
        if kwargs.get('disp_interval') is not None:
            disp_interval = kwargs.get('disp_interval')
        else:
            disp_interval = 10

        if kwargs.get('save_best_result') is not None:
            save_best_result = kwargs.get('save_best_result')
        else:
            save_best_result = False

        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses, loss_terms, x_mu_list, x_sigma_list, z_list_cls, z_list_reg = [], [], [], [], [], []
        best_acc = 0.0
        best_loss = 999999
        for epoch in range(epochs):
            batch_index = self._get_batch_idx(batch_size, self.n)

            optimizer.zero_grad()

            yn_batch = yn[batch_index].to(self.device)
            ys_batch = ys[batch_index].to(self.device)
            # sample_batch = self.sample_latent_variable(yn_batch)

            x = self.sample_latent_variable(yn.to(self.device))
            sample_batch = x[batch_index]
            sample_batch = sample_batch.to(self.device)

            # Calculate loss
            loss, loss_dict = self.elbo(sample_batch, self.x, yn_batch, ys_batch)
            losses.append(loss.item())

            # Back propagate error
            loss.backward()
            optimizer.step()

            if epoch % disp_interval == 0:
                mse_loss = self.update_history_train(yn=yn, elbo_loss=loss.item())
                loss_dict['mse_loss'] = mse_loss
                if yn_test is not None:
                    predicted_ys_test, *_ = self.predict_class(yn_test, ys_test)
                    accuracy_test = check_one_hot_and_get_accuracy(y_true=ys_test, y_predicted=predicted_ys_test)
                    predicted_ys_train, *_ = self.predict_class(yn, ys)
                    accuracy_train = check_one_hot_and_get_accuracy(y_true=ys, y_predicted=predicted_ys_train)
                    mse_loss_test = self.update_history_test(yn_test=yn_test, elbo_loss=loss.item())
                    print(
                        f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, MSE: {mse_loss}, "
                        f"Train Accuracy: {accuracy_train} Test Accuracy: {accuracy_test}")
                    loss_dict['accuracy_train'] = accuracy_train
                    loss_dict['accuracy_test'] = accuracy_test
                    loss_dict['mse_loss'] = mse_loss
                    loss_dict['total_loss'] = loss.item()

                    if save_best_result is True:
                        if accuracy_test >= best_acc and loss.item() < best_loss:
                            best_acc = accuracy_test
                            best_loss = loss.item()
                            self.save_wights(path_save=kwargs.get('path_save'))
                            print("model saved!")
                else:
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, MSE: {mse_loss}")
                loss_terms.append(loss_dict)
                if early_stop is not None:
                    if len(losses) > 100:
                        ce = np.abs(losses[-1] - np.mean(losses[-100:])) / np.abs(np.mean(losses[-100:]))
                        if ce < early_stop:
                            print("Early stop")
                            break

        combined_dict = dicts_to_dict_of_lists(loss_terms)
        if show_plot is True:
            plot_loss(combined_dict)
            plot_loss(losses)

        return losses, combined_dict, self.history_train

    def predict_class(self, yn_test, ys_test, learning_rate=0.01, epochs=100, batch_size=100, early_stop=None,
                      verbos=None, monitor_mse=False, **kwargs):
        if yn_test.shape[1] != self.d:
            raise ValueError(f"yn_test should be of size [num_test, {self.d}]")

        self.n_test = yn_test.shape[0]

        x_test_mu, x_test_logvar = self.x.encode(yn_test.to(self.device))
        predictions, *_ = self.classify_x(x_test_mu)
        loss_terms = None
        return predictions, self.history_test, loss_terms

    def update_history_train(self, yn, elbo_loss):
        x_mu, x_sigma = self.x.encode(yn.to(self.device))
        x_mu = x_mu.cpu().detach().numpy()
        x_sigma = x_sigma.cpu().detach().numpy()

        if self.use_gpytorch is False:
            self.history_train['elbo_loss'].append(elbo_loss)
            self.history_train['x_mu_list'].append(x_mu.copy())
            self.history_train['x_sigma_list'].append(x_sigma.copy())
            self.history_train['z_list_cls'].append(self.inducing_inputs_cls.cpu().detach().numpy().copy())
            self.history_train['z_list_reg'].append(self.inducing_inputs_reg.cpu().detach().numpy().copy())
        else:
            self.history_train['x_mu_list'].append(x_mu.copy())
            self.history_train['x_sigma_list'].append(x_sigma.copy())
            self.history_train['z_list_cls'].append(self.q_f_cls.inducing_points.cpu().detach().numpy().copy())
            self.history_train['z_list_reg'].append(self.q_f_reg.inducing_points.cpu().detach().numpy().copy())

        predicted_yn, predicted_yn_std = self.regress_x(x_mu)
        mse_loss = np.mean(np.square(yn.cpu().detach().numpy() - predicted_yn.cpu().detach().numpy()))
        self.history_train['mse_loss'].append(mse_loss)
        return mse_loss

    def update_history_test(self, yn_test, elbo_loss):
        x_mu, x_sigma = self.x.encode(yn_test.to(self.device))
        x_mu = x_mu.cpu().detach().numpy()
        x_sigma = x_sigma.cpu().detach().numpy()

        if self.use_gpytorch is False:
            self.history_test['elbo_loss'].append(elbo_loss)
            self.history_test['x_mu_list'].append(x_mu.copy())
            self.history_test['x_sigma_list'].append(x_sigma.copy())
            self.history_test['z_list_cls'].append(self.inducing_inputs_cls.cpu().detach().numpy().copy())
            self.history_test['z_list_reg'].append(self.inducing_inputs_reg.cpu().detach().numpy().copy())
        else:
            self.history_test['x_mu_list'].append(x_mu.copy())
            self.history_test['x_sigma_list'].append(x_sigma.copy())
            self.history_test['z_list_cls'].append(self.q_f_cls.inducing_points.cpu().detach().numpy().copy())
            self.history_test['z_list_reg'].append(self.q_f_reg.inducing_points.cpu().detach().numpy().copy())
        predicted_yn, predicted_yn_std = self.regress_x(x_mu)
        mse_loss = np.mean(np.square(yn_test.cpu().detach().numpy() - predicted_yn.cpu().detach().numpy()))
        self.history_test['mse_loss'].append(mse_loss)
        return mse_loss
