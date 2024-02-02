from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from src import *
from src import *
from src import *
from src import ARDRBFKernel, ARDRBFKernelOld
from gp_project_pytorch.data.data_loader import load_dataset

# Set the seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

model_name = 'bgplvm_pytorch'  # 'bgplvm_pytorch' 'bjgplvm' 'bgplvm_scratch'

if model_name == 'bjgplvm':
    """ ========================================================================= """
    """                               Joint     GPLVM                             """
    """ ========================================================================= """
    dataset_names = ['oil', 'wine', 'iris', 'usps']
    yn_train, yn_test, ys_train, ys_test, labels_train, labels_test = load_dataset(dataset_name='iris', test_size=0.1)

    data_dim = yn_train.shape[-1]
    latent_dim = 7
    num_inducing_points = 25
    batch_shape = torch.Size([data_dim])
    num_epochs = 5000
    batch_size = 100
    load_trained_model = False
    use_gpytorch = False

    if use_gpytorch is False:
        kernel_cls = ARDRBFKernel(input_dim=latent_dim)
        kernel_reg = ARDRBFKernel(input_dim=latent_dim)
    else:
        kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))
        kernel_cls = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))

    likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
    likelihood_cls = BernoulliLikelihood()

    model = JointGPLVM_Bayesian(yn_train,
                                kernel_reg=kernel_reg,
                                kernel_cls=kernel_cls,
                                latent_dim=latent_dim,
                                num_inducing_points=num_inducing_points,
                                likelihood_reg=likelihood_reg,
                                likelihood_cls=likelihood_cls,
                                use_gpytorch=use_gpytorch)

    if load_trained_model is False:
        losses = model.train_model(yn=yn_train, ys=ys_train, epochs=num_epochs, batch_size=batch_size)
        model.save_wights()
    else:
        losses = []
        model.load_weights()

    predictions, metrics = model.evaluate(yn_test=yn_test, ys_test=labels_test, epochs=5000)

    if use_gpytorch is False:
        alpha_reg = model.kernel_reg.alpha.detach().numpy()
        alpha_cls = model.kernel_cls.alpha.detach().numpy()
        X = model.x.q_mu.detach().numpy()
        std = model.x.q_sigma.detach().numpy()
    else:
        alpha_reg = 1 / model.kernel_reg.base_kernel.lengthscale.detach().numpy()
        alpha_cls = 1 / model.kernel_cls.base_kernel.lengthscale.detach().numpy()
        X = model.x.q_mu.detach().numpy()
        std = torch.nn.functional.softplus(model.x.q_log_sigma).detach().numpy()

    plot_results_gplvm(X, std, labels=labels_train, losses=losses, inverse_length_scale=alpha_reg,
                       latent_dim=latent_dim)
    plot_results_gplvm(X, std, labels=labels_train, losses=losses, inverse_length_scale=alpha_cls,
                       latent_dim=latent_dim)

    X_test = model.x_test.q_mu.detach().numpy()
    std_test = model.x_test.q_sigma.detach().numpy()
    plot_results_gplvm(X_test, std_test, labels=labels_test, losses=losses, inverse_length_scale=alpha_cls,
                       latent_dim=latent_dim)

    print("Training Finished")
elif model_name == 'bgplvm_scratch':
    """ ========================================================================= """
    """                               Joint     GPLVM                             """
    """ ========================================================================= """
    y, yn_test, ys_train, ys_test, s, labels_test = load_oil_data(test_size=0.01)
    data_dim = y.shape[-1]
    latent_dim = 7
    num_inducing_points = 25
    batch_shape = torch.Size([data_dim])
    use_gpytorch = False

    if use_gpytorch is False:
        kernel_reg = ARDRBFKernel(input_dim=latent_dim)
        # kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))
    else:
        kernel_reg = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim))

    likelihood_reg = GaussianLikelihood(batch_shape=batch_shape)
    likelihood_cls = BernoulliLikelihood()

    model = GPLVM_Bayesian_DS(y,
                              kernel_reg=kernel_reg,
                              latent_dim=latent_dim,
                              num_inducing_points=num_inducing_points,
                              likelihood=likelihood_reg,
                              use_gpytorch=use_gpytorch)
    losses = model.train_model(y=y, epochs=15000, batch_size=100)

    if use_gpytorch is False:
        alpha = model.kernel_reg.alpha.detach().numpy()
        X = model.x.q_mu.detach().numpy()
        std = model.x.q_sigma.detach().numpy()
    else:
        alpha = 1 / model.kernel_reg.base_kernel.lengthscale.detach().numpy()
        X = model.x.q_mu.detach().numpy()
        std = torch.nn.functional.softplus(model.x.q_log_sigma).detach().numpy()

    # std = torch.nn.functional.softplus(model.x.q_log_sigma).detach().numpy()
    # alpha = model.kernel_reg.alpha.detach().numpy()
    plot_results_gplvm(X, std, labels=s, losses=losses, inverse_length_scale=alpha, latent_dim=latent_dim, largest=True)

    print("Training Finished")
elif model_name == 'bgplvm_pytorch':
    """ ========================================================================= """
    """                       Bayesian GPLVM using GPyTorch                       """
    """ ========================================================================= """
    """===================== Bayesian GPLVM using GPyTorch =========================="""
    y, yn_test, ys_train, ys_test, s, labels_test = load_oil_data()
    N = len(y)
    data_dim = y.shape[1]
    latent_dim = 7
    n_inducing = 25
    pca = False

    # Model
    model = bGPLVM(N, data_dim, latent_dim, n_inducing, pca=pca)

    # Likelihood
    likelihood = GaussianLikelihood(batch_shape=model.batch_shape)
    mll = VariationalELBO(likelihood, model, num_data=len(y))

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=0.01)

    loss_list = model.train_model(y, optimizer, mll)

    # vISUALIZATION
    inv_lengthscale = 1 / model.covar_module.base_kernel.lengthscale
    X = model.X.q_mu.detach().numpy()
    std = torch.nn.functional.softplus(model.X.q_log_sigma).detach().numpy()
    alpha = model.kernel_reg.alpha.detach().numpy()
    plot_results_gplvm(X, std, labels=s, losses=loss_list, inverse_length_scale=inv_lengthscale, latent_dim=latent_dim,
                       largest=False)
    print("Training Finished")
elif model_name == 'bgplvm_scratch2':
    """ ========================================================================= """
    """                                    GPLVM                                  """
    """ ========================================================================= """
    y, s = load_oil_data()
    q = 11

    kernel = RBFKernel(input_dim=q)
    ard_kernel = ARDRBFKernelOld(input_dim=q)

    model = GPLVM_Bayesian(y, kernel=ard_kernel, q=q)
    losses = model.train_model(y=y, epochs=10000)

    values, indices = torch.topk(model.kernel.alpha, k=2, largest=True)

    l1 = indices.numpy().flatten()[0]
    l2 = indices.numpy().flatten()[1]

    colors = ['r', 'b', 'g']

    plt.figure(figsize=(20, 8))
    plt.subplot(131)
    X = model.q_mu.detach().numpy()
    std = torch.sqrt(model.q_sigma)
    plt.title('2d latent subspace corresponding to 3 phase oilflow', fontsize='small')
    plt.xlabel('Latent dim 1')
    plt.ylabel('Latent dim 2')

    # Select index of the smallest lengthscales by examining model.covar_module.base_kernel.lengthscales
    for i, label in enumerate(np.unique(s)):
        X_i = X[s == label]
        scale_i = std[s == label].detach().numpy()
        plt.scatter(X_i[:, l1], X_i[:, l2], c=[colors[i]], label=label)
        # plt.errorbar(X_i[:, l1], X_i[:, l2], xerr=scale_i[:, l1], yerr=scale_i[:, l2], label=label, c=colors[i], fmt='none')

    plt.subplot(132)
    plt.bar(np.arange(q), height=model.kernel.alpha.detach().numpy().flatten())
    plt.title('Inverse Lengthscale with SE-ARD kernel', fontsize='small')

    plt.subplot(133)
    plt.plot(losses, label='batch_size=100')
    plt.title('Neg. ELBO Loss', fontsize='small')
    plt.tight_layout()
    plt.show()
elif model_name == 'gplvm_point_estimate':
    """ ========================================================================= """
    """                                    GPLVM                                  """
    """ ========================================================================= """
    """ ================== GPLVM from Scratch ====================="""
    q = 2
    yn_train, yn_test, ys_train, ys_test, labels_train, labels_test = load_dataset(dataset_name='oil', test_size=0.1)
    kernel = ARDRBFKernel(input_dim=q)
    model = GPLVM_point_estimate(yn_train, kernel=kernel, q=q)
    losses = model.train_model(y=yn_train, epochs=500)
    plot_data_2D(model.x.detach(), labels_train)
elif model_name == 'gp_regression_scratch':
    """ ========================================================================= """
    """                               Regression                                  """
    """ ========================================================================= """

    x_train, y_train, x_test, y_test = generate_synthetic_data(N=50)
    kernel = RBFKernel(input_dim=1)

    """ ================= Sparse GPR Use From scratch model =============="""
    inducing_points = torch.linspace(-4, 4, 10)[:, None]
    model = SparseGPRegressionModel(kernel=kernel, z=inducing_points, m=10)

    model.train_model(x_train, y_train, num_epochs=2000)
    mu_star, std_test = model(x_train, y_train, x_test)

    plot_with_confidence(x_train, y_train, x_test, mu_star, std_test, inducing_points=model.z.detach().numpy())
elif model_name == 'gp_sparse_regression_scratch':
    """ ================= Sparse GPR Using GPytorch =============="""
    x_train, y_train, x_test, y_test = generate_synthetic_data(N=50)
    kernel = RBFKernel(input_dim=1)
    selected_indices = np.linspace(0, x_train.shape[0] - 1, 10, dtype=int)
    inducing_points = x_train[selected_indices, 0]
    model_sgpy = SPGPModelGPy(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    losses, likelihood = model_sgpy.train_model(x_train[:, 0], y_train[:, 0], likelihood, training_iter=1000)
    model_sgpy.eval()
    likelihood.eval()

    f_preds = model_sgpy(x_test[:, 0])
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_preds = likelihood(model_sgpy(x_test[:, 0]))

    y_mean = y_preds.mean
    y_std = np.sqrt(y_preds.variance)

    trained_inducing_points = model_sgpy.variational_strategy.inducing_points
    plot_with_confidence(x_train, y_train, x_test, y_mean, y_std, inducing_points=inducing_points)

    """ ================= GPR Use From scratch model =============="""

    model = GPRegressionModel(kernel=kernel)

    model.train_model(x_train, y_train, num_epochs=500)
    mu_star, std_test = model(x_train, y_train, x_test)

    plot_with_confidence(x_train, y_train, x_test, mu_star, std_test)

    """ ================= GPR Use GPY torch =============="""
    training_iter = 1000
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model_gpy = ExactGPModel(x_train[:, 0], y_train[:, 0], likelihood)
    # Find optimal model hyperparameters
    loss = model_gpy.train_model(x_train[:, 0], y_train[:, 0], likelihood, training_iter=100)

    model_gpy.eval()
    likelihood.eval()

    f_preds = model_gpy(x_test[:, 0])
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        y_preds = likelihood(model_gpy(x_test[:, 0]))

    y_mean = y_preds.mean
    y_std = np.sqrt(y_preds.variance)
    plot_with_confidence(x_train, y_train, x_test, y_mean, y_std)
