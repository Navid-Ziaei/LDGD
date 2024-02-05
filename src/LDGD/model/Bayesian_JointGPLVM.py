
from ..model.utils.variational import VariationalDist, VariationalLatentVariable, \
    CholeskeyVariationalDist
from ..visualization import plot_loss
from ..model.utils.variational_strategy import VariationalStrategy2


import json
import math
import torch
import gpytorch
import numpy as np
import torch.nn as nn
from torch import optim
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score


class JointGPLVM_Bayesian(nn.Module):
    def __init__(self, y, kernel_reg, kernel_cls, likelihood_reg, likelihood_cls, latent_dim=2, num_inducing_points=10,
                 num_classes=3,
                 inducing_points=None,
                 use_gpytorch=False,
                 shared_inducing_points=False,
                 use_shared_kernel=False,
                 cls_weight=1.0,
                 reg_weight=1.0):
        super(JointGPLVM_Bayesian, self).__init__()

        self.x_test = None
        self.n_test = None
        self.use_gpytorch = use_gpytorch
        self.initialize_variable = True
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

        self.kernel_reg = kernel_reg
        self.kernel_cls = kernel_cls
        if use_shared_kernel is True:
            self.kernel_cls = kernel_reg

        if isinstance(kernel_reg, gpytorch.kernels.Kernel):
            self.use_gpytorch_kernel = True
        else:
            self.use_gpytorch_kernel = False

        self.likelihood_reg = likelihood_reg
        self.likelihood_cls = likelihood_cls

        self.mean_module = gpytorch.means.ZeroMean(ard_num_dims=latent_dim)
        self.jitter = 1e-4

        self.n = y.shape[0]
        self.m = num_inducing_points
        self.d = y.shape[1]  # Number of featurea
        self.q = latent_dim  # Number of hidden space dimensions
        self.log_noise_sigma = nn.Parameter(torch.ones(1) * -2)
        self.k = num_classes
        self.shared_inducing_points = shared_inducing_points

        self.batch_shape_reg = torch.Size([self.d])
        self.batch_shape_cls = torch.Size([self.k])

        torch.manual_seed(42)
        if inducing_points is None:
            if self.shared_inducing_points is True:
                inducing_inputs_cls = nn.Parameter(torch.randn(num_inducing_points, self.q))
                inducing_inputs_reg = nn.Parameter(torch.randn(num_inducing_points, self.q))

                # Replicate inducing_inputs_cls k times and inducing_inputs_reg d times
                self.inducing_inputs_cls = inducing_inputs_cls.repeat(self.k, 1, 1)
                self.inducing_inputs_reg = inducing_inputs_reg.repeat(self.d, 1, 1)

                # Concatenate along the first dimension to form the final inducing_inputs
                self.inducing_inputs = torch.cat((self.inducing_inputs_reg, self.inducing_inputs_cls), dim=0)
            else:
                self.inducing_inputs_cls = nn.Parameter(torch.randn(self.k, num_inducing_points, self.q))
                self.inducing_inputs_reg = nn.Parameter(torch.randn(self.d, num_inducing_points, self.q))

                self.inducing_inputs = torch.cat((self.inducing_inputs_reg, self.inducing_inputs_cls), dim=0)
                # self.inducing_inputs = nn.Parameter(torch.randn(self.d + self.k, num_inducing_points, self.q))
            self.min_value = 0.001
        else:
            self.inducing_inputs = nn.Parameter(inducing_points)
            self.min_value = 0.001

        x_init = torch.randn(self.n, self.q)
        if self.use_gpytorch is True:
            x_prior_mean = torch.zeros(self.n, self.q)  # shape: N x Q
            self.prior_x = gpytorch.priors.NormalPrior(x_prior_mean, torch.ones_like(x_prior_mean))
            self.x = gpytorch.models.gplvm.VariationalLatentVariable(self.n, self.d, self.q, x_init, self.prior_x)

            self.q_u_reg = gpytorch.variational.CholeskyVariationalDistribution(self.m,
                                                                                batch_shape=self.batch_shape_reg)
            self.q_f_reg = VariationalStrategy2(model=self,
                                                inducing_points=self.inducing_inputs_reg,
                                                variational_distribution=self.q_u_reg,
                                                learn_inducing_locations=True)

            self.q_u_cls = gpytorch.variational.CholeskyVariationalDistribution(self.m,
                                                                                batch_shape=self.batch_shape_cls)
            self.q_f_cls = VariationalStrategy2(model=self,
                                                inducing_points=self.inducing_inputs_cls,
                                                variational_distribution=self.q_u_cls,
                                                learn_inducing_locations=True)
        else:
            self.prior_x = torch.distributions.Normal(torch.zeros(self.n, self.q), torch.ones(self.n, self.q))
            self.x = VariationalLatentVariable(self.n, self.d, self.q, X_init=x_init, prior_x=self.prior_x)

            self.q_u_reg = VariationalDist(num_inducing_points=self.m, batch_shape=self.d)
            self.q_u_cls = VariationalDist(num_inducing_points=self.m, batch_shape=self.k)

            self.q_u_d = CholeskeyVariationalDist(num_inducing_points=self.m, batch_shape=self.d)
            self.q_u_k = CholeskeyVariationalDist(num_inducing_points=self.m, batch_shape=self.k)

            self.log_noise_sigma = nn.Parameter(torch.ones(self.d) * -2)

        self.history_train = {
            'elbo_loss': [],
            'x_mu_list': [],
            'x_sigma_list': [],
            'z_list_cls': [],
            'z_list_reg': [],
            'mse_loss': []
        }
        self.history_test = {}

    @property
    def noise_sigma(self):
        return torch.nn.functional.softplus(self.log_noise_sigma)

    def _get_batch_idx(self, batch_size, n_samples):
        if n_samples < batch_size:
            batch_size = n_samples
        valid_indices = np.arange(n_samples)
        batch_indices = np.random.choice(valid_indices, size=batch_size, replace=False)
        return np.sort(batch_indices)

    def _expand_inputs(self, x, inducing_points):
        """
        Pre-processing step in __call__ to make x the same batch_shape as the inducing points
        """
        batch_shape = torch.broadcast_shapes(inducing_points.shape[:-2], x.shape[:-2])
        inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
        x = x.expand(*batch_shape, *x.shape[-2:])
        return x, inducing_points

    def sample_latent_variable(self):
        return self.x()

    def compute_statistics(self):
        x_samples = self.sample_latent_variable()

        K_nn = self.kernel_reg(x_samples, x_samples)
        K_nm = self.kernel_reg(x_samples, self.inducing_inputs_reg)
        K_mn = K_nm.permute(0, 2, 1)

        self.psi0 = K_nn.diagonal(dim1=1, dim2=2).mean(dim=0).sum()
        self.psi1 = K_nm.mean(dim=0)
        self.psi2 = (K_mn @ K_nm).mean(dim=0)

    def ell_cls_scratch(self, x_samples, ys):
        batch_size = x_samples.shape[0]
        x_samples, z = self._expand_inputs(x_samples, self.inducing_inputs_cls)
        if self.use_gpytorch_kernel is False:
            k_nn_cls = self.kernel_cls(x_samples, x_samples) + \
                       torch.eye(batch_size).unsqueeze(0) * self.jitter  # of size [D, N, N]
            k_mm_cls = self.kernel_cls(self.inducing_inputs_cls,
                                       self.inducing_inputs_cls)  # of size [D, M, M]
            k_mn_cls = self.kernel_cls(self.inducing_inputs_cls, x_samples)
        else:
            k_nn_cls = self.kernel_cls(x_samples, x_samples).evaluate() + torch.eye(batch_size).unsqueeze(
                0) * self.jitter  # of size [D, N, N]
            k_mm_cls = self.kernel_cls(self.inducing_inputs_cls,
                                       self.inducing_inputs_cls).evaluate()  # of size [D, M, M]
            k_mn_cls = self.kernel_cls(self.inducing_inputs_cls, x_samples).evaluate()

        k_mm_cls += torch.eye(k_mm_cls.shape[-1]) * self.jitter

        predictive_dist_cls = self.predictive_distribution(k_nn_cls, k_mm_cls,
                                                           k_mn_cls, variational_mean=self.q_u_cls.mu,
                                                           variational_cov=self.q_u_cls.sigma)
        ell_cls_value = self.likelihood_cls.expected_log_prob(ys.t(), predictive_dist_cls).mean()
        prior_p_u_cls = torch.distributions.MultivariateNormal(torch.zeros(self.m, ), k_mm_cls)
        kl_u_cls = self.q_u_cls.kl(prior_u=prior_p_u_cls)

        return ell_cls_value, kl_u_cls

    def ell_cls_pytorch(self, x_samples, ys):
        batch_size = x_samples.shape[0]

        if self.initialize_variable is True:
            self.q_u_cls.initialize_variational_distribution(self.q_f_cls.prior_distribution)
            self.initialize_variable = False

        inducing_points = self.q_f_cls.inducing_points
        if inducing_points.shape[:-2] != x_samples.shape[:-2]:
            x_samples, inducing_points = self._expand_inputs(x_samples, inducing_points)

        predictive_dist = self.q_f_cls(x=x_samples, mode='cls')

        ell_cls = self.likelihood_cls.expected_log_prob(ys.t(), predictive_dist).sum(-1).div(batch_size)  # of shape [D]
        kl_u_cls = self.q_f_cls.kl_divergence().div(self.n)
        return ell_cls, kl_u_cls

    def ell_reg_scratch(self, x_samples, yn):
        batch_size = x_samples.shape[0]
        x_samples, z = self._expand_inputs(x_samples, self.inducing_inputs_reg)

        if self.use_gpytorch_kernel is False:
            k_nn_reg = self.kernel_reg(x_samples, x_samples) + torch.eye(batch_size).unsqueeze(0) * self.jitter
            k_mm_reg = self.kernel_reg(self.inducing_inputs_reg, self.inducing_inputs_reg)  # of size [D, M, M]
            k_mn_reg = self.kernel_reg(self.inducing_inputs_reg, x_samples)
        else:
            k_nn_reg = self.kernel_reg(x_samples, x_samples).evaluate() + \
                       torch.eye(batch_size).unsqueeze(0) * self.jitter  # of size [D, N, N]
            k_mm_reg = self.kernel_reg(self.inducing_inputs_reg, self.inducing_inputs_reg).evaluate()  # of size[D,M,M]
            k_mn_reg = self.kernel_reg(self.inducing_inputs_reg, x_samples).evaluate()

        k_mm_reg += torch.eye(k_mm_reg.shape[-1]) * self.jitter

        predictive_dist_reg = self.predictive_distribution(k_nn_reg, k_mm_reg,
                                                           k_mn_reg, variational_mean=self.q_u_reg.mu,
                                                           variational_cov=self.q_u_reg.sigma)

        ell_reg = self.likelihood_reg.expected_log_prob(yn.t(), predictive_dist_reg).mean()
        prior_p_u_reg = torch.distributions.MultivariateNormal(torch.zeros(self.m, ), k_mm_reg)
        kl_u_reg = self.q_u_reg.kl(prior_u=prior_p_u_reg)

        return ell_reg, kl_u_reg

    def ell_reg_pytorch(self, x_samples, yn):
        batch_size = x_samples.shape[0]

        if self.initialize_variable is True:
            self.q_u_reg.initialize_variational_distribution(self.q_f_reg.prior_distribution)
            self.initialize_variable = False

        inducing_points = self.q_f_reg.inducing_points
        if inducing_points.shape[:-2] != x_samples.shape[:-2]:
            x, inducing_points = self._expand_inputs(x_samples, inducing_points)

        predictive_dist = self.q_f_reg(x=x)

        ell_reg = self.likelihood_reg.expected_log_prob(yn.t(), predictive_dist).sum(-1).div(batch_size)  # of shape [D]
        kl_u_reg = self.q_f_reg.kl_divergence().div(self.n)

        return ell_reg, kl_u_reg

    def elbo(self, x_samples, x, yn, ys=None):
        if self.use_gpytorch is True:
            if ys is not None:
                ell_cls, kl_u_cls = self.ell_cls_pytorch(x_samples=x_samples, ys=ys)
            ell_reg, kl_u_reg = self.ell_reg_pytorch(x_samples=x_samples, yn=yn)
            kl_x = x._added_loss_terms['x_kl'].loss()
        else:
            if ys is not None:
                ell_cls, kl_u_cls = self.ell_cls_scratch(x_samples=x_samples, ys=ys)
            ell_reg, kl_u_reg = self.ell_reg_scratch(x_samples=x_samples, yn=yn)
            kl_x = x.kl_loss

        loss_reg = ell_reg - kl_u_reg / kl_u_reg.shape[0]
        if ys is not None:
            loss_cls = ell_cls - kl_u_cls / kl_u_cls.shape[0]
            elbo = loss_reg.sum() * self.reg_weight + loss_cls.sum() * self.cls_weight - kl_x
        else:
            elbo = loss_reg.sum() * self.reg_weight - kl_x
        return elbo

    def expected_log_prob_reg(self, target, predictive_dist):
        mean, variance = predictive_dist.mean.t(), predictive_dist.variance.t()
        num_points, num_dims = mean.shape
        # Potentially reshape the noise to deal with the multitask case
        noise = self.noise_sigma

        res = ((target - mean).square() + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)

        return res.sum() / (num_points * num_dims)

    def expected_log_prob_cls(self, target, predictive_dist):
        mean, variance = predictive_dist.mean, predictive_dist.variance

        noise = self._shaped_noise_covar(mean.shape).diagonal(dim1=-1, dim2=-2)
        # Potentially reshape the noise to deal with the multitask case
        noise = noise.view(*noise.shape[:-1], *input.event_shape)

        res = ((target - mean).square() + variance) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)

        return res

    def predictive_distribution(self, k_nn, k_mm, k_mn, variational_mean, variational_cov, whitening_parameters=True):
        # k_mm = LL^T
        L = torch.linalg.cholesky(k_mm)  # torch.cholesky(k_mm, upper=False)
        m_d = variational_mean  # of size [D, M, 1]
        if len(variational_cov.shape) > 2 and variational_cov.shape[1] == variational_cov.shape[2]:
            s_d = variational_cov
        else:
            s_d = torch.diag_embed(variational_cov)  # of size [D, M, M] It's a eye matrix

        prior_dist_co = torch.eye(self.m)

        if whitening_parameters is True:
            # A = A=L^(-1) K_MN  (interp_term)
            interp_term = torch.linalg.solve(L, k_mn)
            # μ_f=A^T m_d^'
            # Σ_f=A^T (S'-I)A
            predictive_mean = (interp_term.transpose(-1, -2) @ m_d.unsqueeze(-1)).squeeze(-1)  # of size [D, N]
            predictive_covar = interp_term.transpose(-1, -2) @ (s_d - prior_dist_co.unsqueeze(0)) @ interp_term
        else:
            # m_f = K_NM K_MM^(-1) m_d
            # sigma_f = K_NM K_MM^(-1) (S_d-K_MM ) K_MM^(-1) K_MN
            interp_term = torch.cholesky_solve(k_mn, L, upper=False)
            predictive_mean = (interp_term.transpose(-1, -2) @ m_d.unsqueeze(-1)).squeeze(-1)  # of size [D, N]
            predictive_covar = interp_term.transpose(-1, -2) @ (s_d - k_mm) @ interp_term
        predictive_covar += k_nn

        return torch.distributions.MultivariateNormal(predictive_mean, predictive_covar)

    def forward(self, X, **kwargs):
        mode = kwargs.get('mode', 'reg')
        mean_x = self.mean_module(X)
        if mode == 'cls':
            covar_x = self.kernel_cls(X, X) + torch.eye(X.shape[1]).unsqueeze(0) * self.jitter
        else:
            covar_x = self.kernel_reg(X, X) + torch.eye(X.shape[1]).unsqueeze(0) * self.jitter
        dist = gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        return dist

    def save_wights(self, path_save=''):
        torch.save(self.state_dict(), path_save + 'model_parameters.pth')

    def load_weights(self, path_save='D:\\Navid\\Projects\\gp_project_pytorch\\', file_name='model_parameters.pth',
                     x_test=None):
        if x_test is not None:
            self.n_test = x_test.shape[0]
            X_init = torch.randn(self.n_test, self.q)
            prior_x_test = torch.distributions.Normal(torch.zeros(self.n_test, self.q), torch.ones(self.n_test, self.q))
            self.x_test = VariationalLatentVariable(self.n_test, self.d, self.q, X_init=X_init, prior_x=prior_x_test)

        self.load_state_dict(torch.load(path_save + file_name))

    def train_model(self, yn, ys, learning_rate=0.01, epochs=100, batch_size=100, early_stop=None):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses, x_mu_list, x_sigma_list, z_list_cls, z_list_reg = [], [], [], [], []
        for epoch in range(epochs):
            batch_index = self._get_batch_idx(batch_size, self.n)
            try:
                optimizer.zero_grad()
                x = self.sample_latent_variable()
                sample_batch = x[batch_index]
                loss = -self.elbo(sample_batch, self.x, yn[batch_index], ys[batch_index])
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:

                    mse_loss = self.update_history_train(yn=yn, elbo_loss=loss.item())
                    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, MSE: {mse_loss}")
                    if early_stop is not None:
                        if len(losses) > 100:
                            ce = np.abs(losses[-1] - np.mean(losses[-100:])) / np.abs(np.mean(losses[-100:]))
                            if ce < early_stop:
                                print("Early stop")
                                break

            except:
                pass

        plot_loss(losses)

        return losses, self.history_train

    def predict_class(self, yn_test, learning_rate=0.01, epochs=100, batch_size=100, early_stop=None):
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
            y_test = torch.zeros(self.n_test, self.k)
            y_test[:, k] = 1
            ys_test_list.append(y_test)

        X_init = torch.randn(self.n_test, self.q)
        if self.use_gpytorch is True:
            x_prior_mean = torch.zeros(self.n_test, self.q)
            prior_x = gpytorch.priors.NormalPrior(x_prior_mean, torch.ones_like(x_prior_mean))
            self.x_test = gpytorch.models.gplvm.VariationalLatentVariable(self.n_test, self.d, self.q, X_init, prior_x)
        else:
            prior_x_test = torch.distributions.Normal(torch.zeros(self.n_test, self.q), torch.ones(self.n_test, self.q))
            self.x_test = VariationalLatentVariable(self.n_test, self.d, self.q, X_init=X_init, prior_x=prior_x_test)

        params_to_optimize = self.x_test.parameters()
        optimizer = optim.Adam(params_to_optimize, lr=learning_rate)

        losses, x_mu_list, x_sigma_list, z_list_reg, z_list_cls = [], [], [], [], []
        for epoch in range(epochs):
            batch_index = self._get_batch_idx(batch_size, self.n_test)
            optimizer.zero_grad()
            x_test = self.x_test()
            x_test_sampled = x_test[batch_index]
            loss = -self.elbo(x_test_sampled, self.x_test, yn_test[batch_index])
            losses.append(loss.item())
            loss.backward(retain_graph=True)
            optimizer.step()

            if epoch % 10 == 0:
                mse_loss = self.update_history_test(yn_test, elbo_loss=loss.item())
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}, MSE: {mse_loss}")
                if early_stop is not None:
                    if len(losses) > 100:
                        ce = np.abs(losses[-1] - np.mean(losses[-100:])) / np.abs(np.mean(losses[-100:]))
                        if ce < early_stop:
                            print("Early stop")
                            break

        predictions, *_ = self.classify_x(self.x_test.q_mu)
        return predictions, self.history_test

    def classify_x(self, x):
        if self.use_gpytorch is True:
            x_test_sampled = x

            if self.inducing_inputs_cls.shape[:-2] != x_test_sampled.shape[:-2]:
                x_samples, inducing_points = self._expand_inputs(x_test_sampled, self.inducing_inputs_cls)

            predictive_dist_cls = self.q_f_cls(x=x_samples, mode='cls')
            predictions = self.likelihood_cls(predictive_dist_cls.mean).mean.argmax(dim=0)
            predictions_probs = self.likelihood_cls(predictive_dist_cls.mean).mean.detach().numpy()
        else:
            batch_size = 500
            predictions, predictions_probs = [], []
            for i in range(0, x.shape[0], batch_size):
                new_batch_size = np.min([i + batch_size, x.shape[0]]) - i
                x_test_sampled = x[i:np.min([i + batch_size, x.shape[0]])]
                x_samples, z = self._expand_inputs(x_test_sampled, self.inducing_inputs_cls)
                if self.use_gpytorch_kernel is False:
                    k_nn_cls = self.kernel_cls(x_samples, x_samples) + \
                               torch.eye(new_batch_size).unsqueeze(0) * self.jitter  # of size [D, N, N]
                    k_mm_cls = self.kernel_cls(self.inducing_inputs_cls, self.inducing_inputs_cls)  # of size [D, M, M]
                    k_mn_cls = self.kernel_cls(self.inducing_inputs_cls, x_samples)
                else:
                    k_nn_cls = self.kernel_cls(x_samples, x_samples).evaluate() + \
                               torch.eye(new_batch_size).unsqueeze(0) * self.jitter  # of size [D, N, N]
                    k_mm_cls = self.kernel_cls(self.inducing_inputs_cls, self.inducing_inputs_cls).evaluate()
                    k_mn_cls = self.kernel_cls(self.inducing_inputs_cls, x_samples).evaluate()

                k_mm_cls += torch.eye(k_mm_cls.shape[-1]) * self.jitter

                predictive_dist_cls = self.predictive_distribution(k_nn_cls, k_mm_cls,
                                                                   k_mn_cls,
                                                                   variational_mean=self.q_u_cls.mu,
                                                                   variational_cov=self.q_u_cls.sigma)
                predictions_probs.append(self.likelihood_cls(predictive_dist_cls.mean).mean.detach().numpy())
                predictions.append(self.likelihood_cls(predictive_dist_cls.mean).mean.argmax(dim=0))
            predictions = np.concatenate(predictions, axis=0)
            predictions_probs = np.concatenate(predictions_probs, axis=1)
        return predictions, predictions_probs

    def regress_x(self, x):
        if self.use_gpytorch is True:
            x_test_sampled = torch.Tensor(x)

            inducing_points = self.inducing_inputs_reg
            if inducing_points.shape[:-2] != x_test_sampled.shape[:-2]:
                x_test_sampled, inducing_points = self._expand_inputs(x_test_sampled, inducing_points)

            predictive_dist_reg = self.q_f_reg(x=x_test_sampled, mode='reg')
            predictions_dist = self.likelihood_reg(predictive_dist_reg.mean)
            predictions_mean = predictions_dist.loc.t()
            predictions_std = predictions_dist.scale.t()
        else:
            batch_size = 500
            predictions_mean, predictions_std = [], []
            for i in range(0, x.shape[0], batch_size):
                new_batch_size = np.min([i + batch_size, x.shape[0]]) - i
                x_test_sampled = torch.Tensor(x[i:np.min([i + batch_size, x.shape[0]])])
                x_samples, z = self._expand_inputs(x_test_sampled, self.inducing_inputs_reg)
                if self.use_gpytorch_kernel is False:
                    k_nn_reg = self.kernel_reg(x_samples, x_samples) + \
                               torch.eye(new_batch_size).unsqueeze(0) * self.jitter  # of size [D, N, N]
                    k_mm_reg = self.kernel_reg(self.inducing_inputs_reg,
                                               self.inducing_inputs_reg)  # of size [D, M, M]
                    k_mn_reg = self.kernel_reg(self.inducing_inputs_reg, x_samples)
                else:
                    k_nn_reg = self.kernel_reg(x_samples, x_samples).evaluate() + \
                               torch.eye(new_batch_size).unsqueeze(0) * self.jitter  # of size [D, N, N]
                    k_mm_reg = self.kernel_reg(self.inducing_inputs_reg,
                                               self.inducing_inputs_reg).evaluate()  # of size [D, M, M]
                    k_mn_reg = self.kernel_reg(self.inducing_inputs_reg, x_samples).evaluate()

                k_mm_reg += torch.eye(k_mm_reg.shape[-1]) * self.jitter

                predictive_dist_reg = self.predictive_distribution(k_nn_reg, k_mm_reg,
                                                                   k_mn_reg,
                                                                   variational_mean=self.q_u_reg.mu,
                                                                   variational_cov=self.q_u_reg.sigma)
                predictions_mean.append(self.likelihood_reg(predictive_dist_reg.mean).loc.t())
                predictions_std.append(self.likelihood_reg(predictive_dist_reg.mean).scale.t())
            predictions_mean = torch.concat(predictions_mean, axis=0)
            predictions_std = torch.concat(predictions_std, axis=1)
        return predictions_mean, predictions_std

    def evaluate(self, yn_test, ys_test, learning_rate=0.01, epochs=100, save_path=None, early_stop=None):
        predictions, history_test = self.predict_class(yn_test, learning_rate=learning_rate, epochs=epochs,
                                                       early_stop=early_stop)
        report = classification_report(y_true=ys_test, y_pred=predictions)
        print(report)
        metrics = {
            'accuracy': accuracy_score(ys_test, predictions),
            'precision': precision_score(ys_test, predictions, average='weighted'),
            'recall': recall_score(ys_test, predictions, average='weighted'),
            'f1_score': f1_score(ys_test, predictions, average='weighted')
        }
        if save_path is not None:
            # Save the report to a text file
            with open(save_path + 'classification_report.txt', "w") as file:
                file.write(report)

            with open(save_path + 'classification_result.json', "w") as file:
                json.dump(metrics, file, indent=2)

        return predictions, metrics, history_test

    def update_history_train(self, yn, elbo_loss):

        if self.use_gpytorch is False:
            x_mu = self.x.q_mu.detach().numpy()
            x_sigma = self.x.q_sigma.detach().numpy()
            self.history_train['elbo_loss'].append(elbo_loss)
            self.history_train['x_mu_list'].append(x_mu.copy())
            self.history_train['x_sigma_list'].append(x_sigma.copy())
            self.history_train['z_list_cls'].append(self.inducing_inputs_cls.detach().numpy().copy())
            self.history_train['z_list_reg'].append(self.inducing_inputs_reg.detach().numpy().copy())
        else:
            x_mu = self.x.q_mu.detach().numpy()
            x_sigma = torch.nn.functional.softplus(self.x.q_log_sigma).detach().numpy()
            self.history_train['x_mu_list'].append(x_mu.copy())
            self.history_train['x_sigma_list'].append(x_sigma.copy())
            self.history_train['z_list_cls'].append(self.q_f_cls.inducing_points.detach().numpy().copy())
            self.history_train['z_list_reg'].append(self.q_f_reg.inducing_points.detach().numpy().copy())

        predicted_yn, predicted_yn_std = self.regress_x(x_mu)
        mse_loss = np.mean(np.square(yn.detach().numpy() - predicted_yn.detach().numpy()))
        self.history_train['mse_loss'].append(mse_loss)
        return mse_loss

    def update_history_test(self, yn_test, elbo_loss):
        if self.use_gpytorch is False:
            x_mu = self.x_test.q_mu.detach().numpy()
            x_sigma = self.x_test.q_sigma.detach().numpy()
            self.history_test['elbo_loss'].append(elbo_loss)
            self.history_test['x_mu_list'].append(x_mu.copy())
            self.history_test['x_sigma_list'].append(x_sigma.copy())
            self.history_test['z_list_cls'].append(self.inducing_inputs_cls.detach().numpy().copy())
            self.history_test['z_list_reg'].append(self.inducing_inputs_reg.detach().numpy().copy())
        else:
            x_mu = self.x_test.q_mu.detach().numpy()
            x_sigma = self.x_test.q_log_sigma.detach().numpy()
            self.history_test['x_mu_list'].append(x_mu.copy())
            self.history_test['x_sigma_list'].append(x_sigma.copy())
            self.history_test['z_list_cls'].append(self.q_f_cls.inducing_points.detach().numpy().copy())
            self.history_test['z_list_reg'].append(self.q_f_reg.inducing_points.detach().numpy().copy())
        predicted_yn, predicted_yn_std = self.regress_x(x_mu)
        mse_loss = np.mean(np.square(yn_test.detach().numpy() - predicted_yn.detach().numpy()))
        self.history_test['mse_loss'].append(mse_loss)
        return mse_loss
