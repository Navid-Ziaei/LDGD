import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import torch
from linear_operator import to_dense
from linear_operator.operators import (
    CholLinearOperator,
    DiagLinearOperator,
    LinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
    SumLinearOperator,
    TriangularLinearOperator,
)
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from torch import Tensor

from gpytorch.variational._variational_strategy import _VariationalStrategy
from gpytorch.variational.cholesky_variational_distribution import CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP
from gpytorch.settings import _linalg_dtype_cholesky, trace_mode
from gpytorch.utils.errors import CachingError
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from gpytorch.utils.warnings import OldVersionWarning

import functools
from abc import ABC, abstractproperty
from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
from linear_operator.operators import LinearOperator
from torch import Tensor

from gpytorch import settings
from gpytorch.distributions import Delta, Distribution, MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import Mean
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy
from gpytorch.module import Module
from gpytorch.utils.memoize import add_to_cache, cached, clear_cache_hook
from gpytorch.variational import _VariationalDistribution


class _BaseExactGP(ExactGP):
    def __init__(
            self,
            train_inputs: Optional[Union[Tensor, Tuple[Tensor, ...]]],
            train_targets: Optional[Tensor],
            likelihood: GaussianLikelihood,
            mean_module: Mean,
            covar_module: Kernel,
    ):
        super().__init__(train_inputs, train_targets, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: Tensor, **kwargs) -> MultivariateNormal:
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


def _add_cache_hook(tsr: Tensor, pred_strat: DefaultPredictionStrategy) -> Tensor:
    if tsr.grad_fn is not None:
        wrapper = functools.partial(clear_cache_hook, pred_strat)
        functools.update_wrapper(wrapper, clear_cache_hook)
        tsr.grad_fn.register_hook(wrapper)
    return tsr


class _VariationalStrategy2(Module, ABC):
    """
    Abstract base class for all Variational Strategies.
    """

    has_fantasy_strategy = False

    def __init__(
            self,
            model: Union[ApproximateGP, "_VariationalStrategy"],
            inducing_points: Tensor,
            variational_distribution: _VariationalDistribution,
            learn_inducing_locations: bool = True,
            jitter_val: Optional[float] = None,
    ):
        super().__init__()

        self._jitter_val = jitter_val

        # Model
        object.__setattr__(self, "model", model)

        # Inducing points
        inducing_points = inducing_points.clone()
        if inducing_points.dim() == 1:
            inducing_points = inducing_points.unsqueeze(-1)
        if learn_inducing_locations:
            cls_repeated = self.inducing_inputs_cls.repeat(self.k, 1, 1)
            self.inducing_inputs_reg = nn.Parameter(torch.randn(self.d, num_inducing_points, self.q))

            self.inducing_inputs = torch.cat((self.inducing_inputs_reg, self.inducing_inputs_cls), dim=0)


            self.register_parameter(name="inducing_points", parameter=torch.nn.Parameter(inducing_points))
        else:
            self.register_buffer("inducing_points", inducing_points)

        # Variational distribution
        self._variational_distribution = variational_distribution
        self.register_buffer("variational_params_initialized", torch.tensor(0))

    def _clear_cache(self) -> None:
        clear_cache_hook(self)

    def _expand_inputs(self, x: Tensor, inducing_points: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Pre-processing step in __call__ to make x the same batch_shape as the inducing points
        """
        batch_shape = torch.broadcast_shapes(inducing_points.shape[:-2], x.shape[:-2])
        inducing_points = inducing_points.expand(*batch_shape, *inducing_points.shape[-2:])
        x = x.expand(*batch_shape, *x.shape[-2:])
        return x, inducing_points

    @property
    def jitter_val(self) -> float:
        if self._jitter_val is None:
            return settings.variational_cholesky_jitter.value(dtype=self.inducing_points.dtype)
        return self._jitter_val

    @jitter_val.setter
    def jitter_val(self, jitter_val: float):
        self._jitter_val = jitter_val

    @abstractproperty
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.prior_distribution` method determines how to compute the
        GP prior distribution of the inducing points, e.g. :math:`p(u) \sim N(\mu(X_u), K(X_u, X_u))`. Most commonly,
        this is done simply by calling the user defined GP prior on the inducing point data directly.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`p( \mathbf u)`
        """
        raise NotImplementedError

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self) -> Distribution:
        return self._variational_distribution()

    def forward(
            self,
            x: Tensor,
            inducing_points: Tensor,
            inducing_values: Tensor,
            variational_inducing_covar: Optional[LinearOperator] = None,
            **kwargs,
    ) -> MultivariateNormal:
        r"""
        The :func:`~gpytorch.variational.VariationalStrategy.forward` method determines how to marginalize out the
        inducing point function values. Specifically, forward defines how to transform a variational distribution
        over the inducing point values, :math:`q(u)`, in to a variational distribution over the function values at
        specified locations x, :math:`q(f|x)`, by integrating :math:`\int p(f|x, u)q(u)du`

        :param x: Locations :math:`\mathbf X` to get the
            variational posterior of the function values at.
        :param inducing_points: Locations :math:`\mathbf Z` of the inducing points
        :param inducing_values: Samples of the inducing function values :math:`\mathbf u`
            (or the mean of the distribution :math:`q(\mathbf u)` if q is a Gaussian.
        :param variational_inducing_covar: If
            the distribuiton :math:`q(\mathbf u)` is
            Gaussian, then this variable is the covariance matrix of that Gaussian.
            Otherwise, it will be None.

        :rtype: :obj:`~gpytorch.distributions.MultivariateNormal`
        :return: The distribution :math:`q( \mathbf f(\mathbf X))`
        """
        raise NotImplementedError

    def kl_divergence(self) -> Tensor:
        r"""
        Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        and the prior inducing distribution :math:`p(\mathbf u)`.
        """
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence

    @cached(name="amortized_exact_gp")
    def amortized_exact_gp(
            self, mean_module: Optional[Module] = None, covar_module: Optional[Module] = None
    ) -> ExactGP:
        mean_module = self.model.mean_module if mean_module is None else mean_module
        covar_module = self.model.covar_module if covar_module is None else covar_module

        with torch.no_grad():
            # from here on down, we refer to the inducing points as pseudo_inputs
            pseudo_target_covar, pseudo_target_mean = self.pseudo_points
            pseudo_inputs = self.inducing_points.detach()
            if pseudo_inputs.ndim < pseudo_target_mean.ndim:
                pseudo_inputs = pseudo_inputs.expand(*pseudo_target_mean.shape[:-2], *pseudo_inputs.shape)
            # TODO: add flag for conditioning into SGPR after building fantasy strategy for SGPR
            new_covar_module = deepcopy(covar_module)

            # update inducing mean if necessary
            pseudo_target_mean = pseudo_target_mean.squeeze() + mean_module(pseudo_inputs)

            inducing_exact_model = _BaseExactGP(
                pseudo_inputs,
                pseudo_target_mean,
                mean_module=deepcopy(mean_module),
                covar_module=new_covar_module,
                likelihood=deepcopy(self.model.likelihood),
            )

            # now fantasize around this model
            # as this model is new, we need to compute a posterior to construct the prediction strategy
            # which uses the likelihood pseudo caches
            faked_points = torch.randn(
                *pseudo_target_mean.shape[:-2],
                1,
                pseudo_inputs.shape[-1],
                device=pseudo_inputs.device,
                dtype=pseudo_inputs.dtype,
            )
            inducing_exact_model.eval()
            _ = inducing_exact_model(faked_points)

            # then we overwrite the likelihood to take into account the multivariate normal term
            pred_strat = inducing_exact_model.prediction_strategy
            pred_strat._memoize_cache = {}
            with torch.no_grad():
                updated_lik_train_train_covar = pred_strat.train_prior_dist.lazy_covariance_matrix + pseudo_target_covar
                pred_strat.lik_train_train_covar = updated_lik_train_train_covar

            # do the mean cache because the mean cache doesn't solve against lik_train_train_covar
            train_mean = inducing_exact_model.mean_module(*inducing_exact_model.train_inputs)
            train_labels_offset = (inducing_exact_model.prediction_strategy.train_labels - train_mean).unsqueeze(-1)
            mean_cache = updated_lik_train_train_covar.solve(train_labels_offset).squeeze(-1)
            mean_cache = _add_cache_hook(mean_cache, inducing_exact_model.prediction_strategy)
            add_to_cache(pred_strat, "mean_cache", mean_cache)
            # TODO: check to see if we need to do the covar_cache?

            inducing_exact_model.prediction_strategy = pred_strat
        return inducing_exact_model

    def pseudo_points(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("Each variational strategy must implement its own pseudo points method")

    def get_fantasy_model(
            self,
            inputs: Tensor,
            targets: Tensor,
            mean_module: Optional[Module] = None,
            covar_module: Optional[Module] = None,
            **kwargs,
    ) -> ExactGP:
        r"""
        Performs the online variational conditioning (OVC) strategy of Maddox et al, '21 to return
        an exact GP model that incorporates the inputs and targets alongside the variational model's inducing
        points and targets.

        Currently, instead of directly updating the variational parameters (and inducing points), we instead
        return an ExactGP model rather than an updated variational GP model. This is done primarily for
        numerical stability.

        Unlike the ExactGP's call for get_fantasy_model, we enable options for mean_module and covar_module
        that allow specification of the mean / covariance. We expect that either the mean and covariance
        modules are attributes of the model itself called mean_module and covar_module respectively OR that you
        pass them into this method explicitly.

        :param inputs: (`b1 x ... x bk x m x d` or `f x b1 x ... x bk x m x d`) Locations of fantasy
            observations.
        :param targets: (`b1 x ... x bk x m` or `f x b1 x ... x bk x m`) Labels of fantasy observations.
        :param mean_module: torch module describing the mean function of the GP model. Optional if
            `mean_module` is already an attribute of the variational GP.
        :param covar_module: torch module describing the covariance function of the GP model. Optional
            if `covar_module` is already an attribute of the variational GP.
        :return: An `ExactGP` model with `k + m` training examples, where the `m` fantasy examples have been added
            and all test-time caches have been updated. We assume that there are `k` inducing points in this variational
            GP. Note that we return an `ExactGP` rather than a variational GP.

        Reference: "Conditioning Sparse Variational Gaussian Processes for Online Decision-Making,"
            Maddox, Stanton, Wilson, NeurIPS, '21
            https://papers.nips.cc/paper/2021/hash/325eaeac5bef34937cfdc1bd73034d17-Abstract.html
        """

        # currently, we only support fantasization for CholeskyVariationalDistribution and
        # whitened / unwhitened variational strategies
        if not self.has_fantasy_strategy:
            raise NotImplementedError(
                "No fantasy model support for ",
                self.__name__,
                ". Only VariationalStrategy and UnwhitenedVariationalStrategy are currently supported.",
            )
        if not isinstance(self.model.likelihood, GaussianLikelihood):
            raise NotImplementedError(
                "No fantasy model support for ",
                self.model.likelihood,
                ". Only GaussianLikelihoods are currently supported.",
            )
        # we assume that either the user has given the model a mean_module and a covar_module
        # or that it will be passed into the get_fantasy_model function. we check for these.
        if mean_module is None:
            mean_module = getattr(self.model, "mean_module", None)
            if mean_module is None:
                raise ModuleNotFoundError(
                    "Either you must provide a mean_module as input to get_fantasy_model",
                    "or it must be an attribute of the model called mean_module.",
                )
        if covar_module is None:
            covar_module = getattr(self.model, "covar_module", None)
            if covar_module is None:
                # raise an error
                raise ModuleNotFoundError(
                    "Either you must provide a covar_module as input to get_fantasy_model",
                    "or it must be an attribute of the model called covar_module.",
                )

        # first we construct an exact model over the inducing points with the inducing covariance
        # matrix
        inducing_exact_model = self.amortized_exact_gp(mean_module=mean_module, covar_module=covar_module)

        # then we update this model by adding in the inputs and pseudo targets
        # finally we fantasize wrt targets
        fantasy_model = inducing_exact_model.get_fantasy_model(inputs, targets, **kwargs)
        fant_pred_strat = fantasy_model.prediction_strategy

        # first we update the lik_train_train_covar
        # do the mean cache again because the mean cache resets the likelihood forward
        train_mean = fantasy_model.mean_module(*fantasy_model.train_inputs)
        train_labels_offset = (fant_pred_strat.train_labels - train_mean).unsqueeze(-1)
        fantasy_lik_train_root_inv = fant_pred_strat.lik_train_train_covar.root_inv_decomposition()
        mean_cache = fantasy_lik_train_root_inv.matmul(train_labels_offset).squeeze(-1)
        mean_cache = _add_cache_hook(mean_cache, fant_pred_strat)
        add_to_cache(fant_pred_strat, "mean_cache", mean_cache)
        # TODO: should we update the covar_cache?

        fantasy_model.prediction_strategy = fant_pred_strat
        return fantasy_model

    def __call__(self, x: Tensor, prior: bool = False, **kwargs) -> MultivariateNormal:
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x, **kwargs)

        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()
        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            prior_dist = self.prior_distribution
            self._variational_distribution.initialize_variational_distribution(prior_dist)
            self.variational_params_initialized.fill_(1)

        # Ensure inducing_points and x are the same size
        inducing_points = self.inducing_points
        if inducing_points.shape[:-2] != x.shape[:-2]:
            x, inducing_points = self._expand_inputs(x, inducing_points)

        # Get p(u)/q(u)
        variational_dist_u = self.variational_distribution

        # Get q(f)
        if isinstance(variational_dist_u, MultivariateNormal):
            return super().__call__(
                x,
                inducing_points,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                **kwargs,
            )
        elif isinstance(variational_dist_u, Delta):
            return super().__call__(
                x, inducing_points, inducing_values=variational_dist_u.mean, variational_inducing_covar=None, **kwargs
            )
        else:
            raise RuntimeError(
                f"Invalid variational distribuition ({type(variational_dist_u)}). "
                "Expected a multivariate normal or a delta distribution."
            )


def _ensure_updated_strategy_flag_set(
        state_dict: Dict[str, Tensor],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: Iterable[str],
        unexpected_keys: Iterable[str],
        error_msgs: Iterable[str],
):
    device = state_dict[list(state_dict.keys())[0]].device
    if prefix + "updated_strategy" not in state_dict:
        state_dict[prefix + "updated_strategy"] = torch.tensor(False, device=device)
        warnings.warn(
            "You have loaded a variational GP model (using `VariationalStrategy`) from a previous version of "
            "GPyTorch. We have updated the parameters of your model to work with the new version of "
            "`VariationalStrategy` that uses whitened parameters.\nYour model will work as expected, but we "
            "recommend that you re-save your model.",
            OldVersionWarning,
        )


class VariationalStrategy2(_VariationalStrategy):
    r"""
    The standard variational strategy, as defined by `Hensman et al. (2015)`_.
    This strategy takes a set of :math:`m \ll n` inducing points :math:`\mathbf Z`
    and applies an approximate distribution :math:`q( \mathbf u)` over their function values.
    (Here, we use the common notation :math:`\mathbf u = f(\mathbf Z)`.
    The approximate function distribution for any abitrary input :math:`\mathbf X` is given by:

    .. math::

        q( f(\mathbf X) ) = \int p( f(\mathbf X) \mid \mathbf u) q(\mathbf u) \: d\mathbf u

    This variational strategy uses "whitening" to accelerate the optimization of the variational
    parameters. See `Matthews (2017)`_ for more info.

    :param model: Model this strategy is applied to.
        Typically passed in when the VariationalStrategy is created in the
        __init__ method of the user defined model.
    :param inducing_points: Tensor containing a set of inducing
        points to use for variational inference.
    :param variational_distribution: A
        VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    :param learn_inducing_locations: (Default True): Whether or not
        the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
        parameters of the model).
    :param jitter_val: Amount of diagonal jitter to add for Cholesky factorization numerical stability

    .. _Hensman et al. (2015):
        http://proceedings.mlr.press/v38/hensman15.pdf
    .. _Matthews (2017):
        https://www.repository.cam.ac.uk/handle/1810/278022
    """

    def __init__(
            self,
            model: ApproximateGP,
            inducing_points: Tensor,
            variational_distribution: _VariationalDistribution,
            learn_inducing_locations: bool = True,
            jitter_val: Optional[float] = None,
    ):
        super().__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations, jitter_val=jitter_val
        )
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)
        self.has_fantasy_strategy = True

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar: LinearOperator) -> TriangularLinearOperator:
        L = psd_safe_cholesky(to_dense(induc_induc_covar).type(_linalg_dtype_cholesky.value()))
        return TriangularLinearOperator(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self) -> MultivariateNormal:
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLinearOperator(ones))
        return res

    @property
    @cached(name="pseudo_points_memo")
    def pseudo_points(self) -> Tuple[Tensor, Tensor]:
        # TODO: have var_mean, var_cov come from a method of _variational_distribution
        # while having Kmm_root be a root decomposition to enable CIQVariationalDistribution support.

        # retrieve the variational mean, m and covariance matrix, S.
        if not isinstance(self._variational_distribution, CholeskyVariationalDistribution):
            raise NotImplementedError(
                "Only CholeskyVariationalDistribution has pseudo-point support currently, ",
                "but your _variational_distribution is a ",
                self._variational_distribution.__name__,
            )

        var_cov_root = TriangularLinearOperator(self._variational_distribution.chol_variational_covar)
        var_cov = CholLinearOperator(var_cov_root)
        var_mean = self.variational_distribution.mean
        if var_mean.shape[-1] != 1:
            var_mean = var_mean.unsqueeze(-1)

        # compute R = I - S
        cov_diff = var_cov.add_jitter(-1.0)
        cov_diff = -1.0 * cov_diff

        # K^{1/2}
        Kmm = self.model.covar_module(self.inducing_points)
        Kmm_root = Kmm.cholesky()

        # D_a = (S^{-1} - K^{-1})^{-1} = S + S R^{-1} S
        # note that in the whitened case R = I - S, unwhitened R = K - S
        # we compute (R R^{T})^{-1} R^T S for stability reasons as R is probably not PSD.
        eval_var_cov = var_cov.to_dense()
        eval_rhs = cov_diff.transpose(-1, -2).matmul(eval_var_cov)
        inner_term = cov_diff.matmul(cov_diff.transpose(-1, -2))
        # TODO: flag the jitter here
        inner_solve = inner_term.add_jitter(self.jitter_val).solve(eval_rhs, eval_var_cov.transpose(-1, -2))
        inducing_covar = var_cov + inner_solve

        inducing_covar = Kmm_root.matmul(inducing_covar).matmul(Kmm_root.transpose(-1, -2))

        # mean term: D_a S^{-1} m
        # unwhitened: (S - S R^{-1} S) S^{-1} m = (I - S R^{-1}) m
        rhs = cov_diff.transpose(-1, -2).matmul(var_mean)
        # TODO: this jitter too
        inner_rhs_mean_solve = inner_term.add_jitter(self.jitter_val).solve(rhs)
        pseudo_target_mean = Kmm_root.matmul(inner_rhs_mean_solve)

        # ensure inducing covar is psd
        # TODO: make this be an explicit root decomposition
        try:
            pseudo_target_covar = CholLinearOperator(inducing_covar.add_jitter(self.jitter_val).cholesky()).to_dense()
        except NotPSDError:
            from linear_operator.operators import DiagLinearOperator

            evals, evecs = torch.linalg.eigh(inducing_covar)
            pseudo_target_covar = (
                evecs.matmul(DiagLinearOperator(evals + self.jitter_val)).matmul(evecs.transpose(-1, -2)).to_dense()
            )

        return pseudo_target_covar, pseudo_target_mean

    def forward(
            self,
            x: Tensor,
            inducing_points: Tensor,
            inducing_values: Tensor,
            variational_inducing_covar: Optional[LinearOperator] = None,
            **kwargs,
    ) -> MultivariateNormal:
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:]
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter(self.jitter_val)
        induc_data_covar = full_covar[..., :num_induc, num_induc:].to_dense()
        data_data_covar = full_covar[..., num_induc:, num_induc:]

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar)
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.solve(induc_data_covar.type(_linalg_dtype_cholesky.value())).to(full_inputs.dtype)

        # Compute the mean of q(f)
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)
        if variational_inducing_covar is not None:
            middle_term = SumLinearOperator(variational_inducing_covar, middle_term)

        if trace_mode.on():
            predictive_covar = (
                    data_data_covar.add_jitter(self.jitter_val).to_dense()
                    + interp_term.transpose(-1, -2) @ middle_term.to_dense() @ interp_term
            )
        else:
            predictive_covar = SumLinearOperator(
                data_data_covar.add_jitter(self.jitter_val),
                MatmulLinearOperator(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x: Tensor, prior: bool = False, **kwargs) -> MultivariateNormal:
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u)
                prior_function_dist = self(self.inducing_points, prior=True)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter(self.jitter_val))

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                if isinstance(variational_dist, MultivariateNormal):
                    mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).type(_linalg_dtype_cholesky.value())
                    whitened_mean = L.solve(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                    covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.to_dense()
                    covar_root = covar_root.type(_linalg_dtype_cholesky.value())
                    whitened_covar = RootLinearOperator(L.solve(covar_root).to(variational_dist.loc.dtype))
                    whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                    self._variational_distribution.initialize_variational_distribution(
                        whitened_variational_distribution
                    )

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, **kwargs)
