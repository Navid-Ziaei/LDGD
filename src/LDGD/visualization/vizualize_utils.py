import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import stats, signal
import matplotlib
import seaborn
from scipy.stats import multivariate_normal
import pandas as pd
import GPy
import torch
import matplotlib.pyplot as plt

seaborn.set(style="white", color_codes=True)
AXIS_LABEL_FONTSIZE = 32
TICKS_LABEL_FONTSIZE = 28
LEGEND_FONTSIZE = 24


def plot_2d_scatter(X, y, save_path=None, ax=None, fig=None):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(12, 8))

    ax.scatter(X[y == 0, 0], X[y == 0, 1],
               c='r', s=40, alpha=1,
               edgecolor='r', label='Class 1')  # edgecolor adds a border to the markers
    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               c='b', s=40,
               edgecolor='b', label='Class 2')
    # Adding labels with larger font size
    ax.set_xlabel('$X_1$', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel('$X_2$', fontsize=AXIS_LABEL_FONTSIZE)

    # Setting tick label sizes
    ax.tick_params(axis='both', labelsize=TICKS_LABEL_FONTSIZE)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    # Optional: Add grid
    ax.grid(True, linestyle='--', alpha=0.5)

    # Optional: Set aspect ratio
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]

    # Calculate the aspect ratio
    # The figsize tuple contains the figure's width and height in inches.
    # The aspect ratio should consider the figure's dimensions to scale the axes correctly.
    # fig_aspect_ratio = fig.get_figwidth() / fig.get_figheight()
    data_aspect_ratio = (x_range / y_range)  # * fig_aspect_ratio
    ax.set_aspect(data_aspect_ratio, adjustable='box')

    # Optional: Add legend
    # If you have a legend to add, uncomment and modify the following line
    ax.legend(fontsize=LEGEND_FONTSIZE)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path + "dataset.png")
        fig.savefig(save_path + "dataset.svg")


def plot_scatter_gplvm(X, labels, l1=0, l2=1, ax=None, colors=['r', 'b', 'g'], show_errorbars=True, std=None):
    if ax is None:
        plt.figure(figsize=(7, 8))
        ax = plt.subplot(131)

    # ax1.set_title('2D Latent Subspace Corresponding to 3 Phase Oilflow', fontsize=32)
    ax.set_xlabel(f'Latent Dimension {l1 + 1}', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(f'Latent Dimension {l2 + 1}', fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICKS_LABEL_FONTSIZE)
    # Applying consistent spines format
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    # Applying consistent grid format
    for i, label in enumerate(np.unique(labels)):
        X_i = X[labels == label]

        ax.scatter(X_i[:, l1], X_i[:, l2], c=colors[i], label=f'Class {label + 1}', s=40, edgecolor=colors[i], alpha=1)
        if show_errorbars is True and std is not None:
            scale_i = std[labels == label]
            ax.errorbar(X_i[:, l1], X_i[:, l2], xerr=scale_i[:, l1], yerr=scale_i[:, l2], fmt='none', ecolor=colors[i],
                        alpha=0.5, label=f'Confidence {label + 1}')
    ax.legend(fontsize=LEGEND_FONTSIZE)


def plot_ARD_gplvm(latent_dim, inverse_length_scale, ax=None):
    if ax is None:
        fig, ax = plt.subplots(132)
    ax.bar(np.arange(latent_dim), height=inverse_length_scale.flatten())
    ax.set_xlabel("ARD Coefficients", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Value", fontsize=AXIS_LABEL_FONTSIZE)
    # ax2.set_title('Inverse Lengthscale with SE-ARD Kernel', fontsize=32)
    ax.tick_params(axis='both', labelsize=TICKS_LABEL_FONTSIZE)
    # Applying consistent spines format
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)


def plot_loss_gplvm(losses, ax=None):
    if ax is None:
        fig, ax = plt.subplots(132)
    ax.plot(losses[10:], label='batch_size=100')
    ax.set_xlabel("Iteration", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("ELBO Loss", fontsize=AXIS_LABEL_FONTSIZE)
    # ax.set_title('Neg. ELBO Loss', fontsize=32)
    ax.tick_params(axis='both', labelsize=TICKS_LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    # Applying consistent spines format
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)


def plot_results_gplvm(X, std, labels, losses, inverse_length_scale, latent_dim, largest=True, save_path=None,
                       file_name='gplvm_result', show_errorbars=True):
    values, indices = torch.topk(torch.tensor(inverse_length_scale), k=2, largest=largest)
    l1 = indices.numpy().flatten()[0]
    l2 = indices.numpy().flatten()[1]

    colors = ['r', 'b', 'g']

    plt.figure(figsize=(20, 8))

    # 2D Latent Subspace Plot
    ax1 = plt.subplot(131)
    plot_scatter_gplvm(X, labels, l1=l1, l2=l2, ax=ax1, colors=['r', 'b', 'g'], show_errorbars=show_errorbars, std=std)

    # ARD Coefficients Plot
    ax2 = plt.subplot(132)
    plot_ARD_gplvm(latent_dim, inverse_length_scale, ax=ax2)

    # Neg. ELBO Loss Plot
    ax3 = plt.subplot(133)
    plot_loss_gplvm(losses, ax=ax3)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}{file_name}.png')
        plt.savefig(f'{save_path}{file_name}.svg')
    else:
        plt.show()


def plot_heatmap(x, labels, model, alpha, x_std=None, cmap='winter', range_scale=1.2, save_path=None,
                 file_name='latent_heatmap', inducing_points=None, ind_point=0, ax1=None, fig=None, show_plot=False):
    values, indices = torch.topk(torch.tensor(alpha), k=2, largest=True)
    l1 = indices.numpy().flatten()[0]
    l2 = indices.numpy().flatten()[1]

    # Define the range and resolution of your grid
    x_min, x_max = np.min([0, x[:, l1].min() * range_scale]), x[:, l1].max() * range_scale  # adjust as needed
    y_min, y_max = np.min([0, x[:, l2].min() * range_scale]), x[:, l2].max() * range_scale  # adjust as needed

    if inducing_points is not None:
        z_reg, z_cls = inducing_points
        z_reg1_min, z_reg1_max = np.min([0, z_reg[..., l1].min() * range_scale]), z_reg[..., l1].max() * range_scale
        z_reg2_min, z_reg2_max = np.min([0, z_reg[..., l2].min() * range_scale]), z_reg[..., l2].max() * range_scale
        z_cls1_min, z_cls1_max = np.min([0, z_cls[..., l1].min() * range_scale]), z_cls[..., l1].max() * range_scale
        z_cls2_min, z_cls2_max = np.min([0, z_cls[..., l2].min() * range_scale]), z_cls[..., l2].max() * range_scale

        z1_min, z1_max = np.min([z_reg1_min, z_cls1_min]), np.max([z_reg1_max, z_cls1_max])
        z2_min, z2_max = np.min([z_reg2_min, z_cls2_min]), np.max([z_reg2_max, z_cls2_max])

        x_min, x_max = np.min([z1_min, x_min]), np.max([z1_max, x_max])
        y_min, y_max = np.min([z2_min, y_min]), np.max([z2_max, y_max])

    resolution = 150  # number of points along each axis

    # Create a grid of points
    x_values = torch.linspace(x_min, x_max, resolution)
    y_values = torch.linspace(y_min, y_max, resolution)
    xx, yy = torch.meshgrid(x_values, y_values, indexing='xy')

    X_grid = torch.zeros((yy.ravel().shape[0], x.shape[-1]))
    X_grid[:, l1] = xx.ravel()
    X_grid[:, l2] = yy.ravel()

    prediction, predictions_probs = model.classify_x(X_grid)
    prob_matrix = predictions_probs[1, :].reshape(resolution, resolution)

    if ax1 is None:
        fig, ax1 = plt.subplots(1, figsize=(8, 8))

    data_aspect_ratio = (y_max - y_min) / (x_max - x_min)

    # Use imshow to plot the heatmap
    img = ax1.imshow(prob_matrix,
                     extent=[x_values.min().item(), x_values.max().item(), y_values.min().item(),
                             y_values.max().item()],
                     origin='lower', cmap=cmap, alpha=0.7)

    # Adding a colorbar
    cbar = fig.colorbar(img, ax=ax1, label='Probabilities', pad=0.04, fraction=0.0458)
    cbar.ax.set_ylabel('Probabilities', fontsize=24)  # Set the colorbar label fontsize
    cbar.ax.tick_params(labelsize=20)  # Set the colorbar tick label fontsize

    # Extracting the relevant dimensions from x_mu_list_test
    color_list = ['black', 'white', 'y']
    label_list = ['class1', 'class2', 'class3']
    labels_name = [f'class{i}' for i in labels]
    for idx, label in enumerate(np.unique(labels)):
        ax1.scatter(x[labels == label, l1], x[labels == label, l2], c=color_list[idx], label=label_list[idx])
        if x_std is not None:
            plt.errorbar(x[labels == label, l1], x[labels == label, l2],
                         xerr=x_std[labels == label, l1],
                         yerr=x_std[labels == label, l2],
                         c='red', label='Error bar',
                         fmt='none')

    ax1.contour(xx, yy, prob_matrix, levels=[0.5], colors='k', label='Decision boundary')  # 'k' sets the color to black

    if inducing_points is not None:
        ax1.scatter(z_cls[..., l1].ravel(),
                    z_cls[..., l2].ravel(),
                    c='r', marker='x', label='Classification inducing points')
        ax1.scatter(z_reg[..., l1].ravel(),
                    z_reg[..., l2].ravel(),
                    c='blue', marker='x', label='Regression inducing points')

    # Setting labels with larger font size (like the previous figure)
    ax1.set_xlabel('x1', fontsize=28)
    ax1.set_ylabel('x2', fontsize=28)

    # Setting tick label sizes (like the previous figure)
    ax1.tick_params(axis='both', labelsize=24)

    ax1.legend()

    # Adjusting spines (like the previous figure)
    for spine in ax1.spines.values():
        spine.set_linewidth(1)

    # Optional: Set aspect ratio (like the previous figure)
    ax1.set_aspect('equal', adjustable='box')

    # Ensuring the layout is tight and no overlaps

    ax1.set_aspect(aspect=1 / data_aspect_ratio, adjustable='box')
    plt.tight_layout()

    # Display the plot
    if save_path is not None:
        fig.savefig(save_path + file_name + '.png')
        fig.savefig(save_path + file_name + '.svg')
    if show_plot is True:
        plt.show()

    return fig


def visualize_gp_synthetic_data(X, Y, sample_idx=0):
    # Assuming X has shape TxQ and Y has shape NxDxT

    # Set up subplots
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 15))

    # Plot kernels
    t = np.linspace(0, 1, X.shape[0])
    for q in range(X.shape[1]):
        axs[0].plot(t, X[:, q], label=f'Latent Variable {q + 1}')
    axs[0].set_title('Latent Variables')
    axs[0].legend()

    # Plot neural data for given sample index
    for d in range(Y.shape[1]):
        axs[1].plot(t, Y[sample_idx, d, :], label=f'Channel {d + 1}')
    axs[1].set_title(f'Neural Data for Sample {sample_idx}')
    axs[1].legend()

    # Plotting GP outputs (mean and confidence intervals) for each stimulus
    kernel = GPy.kern.Matern32(input_dim=2)
    for s in [0, 1]:
        gp = GPy.models.GPRegression(X, Y[sample_idx, ...].T, kernel)
        mu, C = gp.predict(X)
        axs[2 + s].plot(t, mu, 'b-', lw=2)
        axs[2 + s].fill_between(t, mu[:, 0] - 1.96 * np.sqrt(C[:, 0]), mu[:, 0] + 1.96 * np.sqrt(C[:, 0]), color='blue',
                                alpha=0.2)
        axs[2 + s].set_title(f'GP Output for Stimulus {s + 1}')

    plt.tight_layout()
    plt.show()


def plot_gaussian(erp_features, ch=1, t1=0, t2=10, ax=None, title=None):
    # Fit a Gaussian distribution to the data
    x = erp_features[:, ch, [t1, t2]]
    # x = x[labels==1]
    # x = x.reshape(-1,2)[:5000]
    mu = np.mean(x, axis=0)
    sigma = np.cov(x.T)
    mvn = multivariate_normal(mu, sigma)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if title is not None:
        ax.set_title(title)
    # Create a grid of points to evaluate the Gaussian distribution
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    pos = np.dstack((xx, yy))

    # Evaluate the Gaussian distribution at each point on the grid
    z = mvn.pdf(pos)

    ax.scatter(x[:, 0], x[:, 1], alpha=0.5)
    ax.contour(xx, yy, z)
    ax.set_xlabel('X(t1)')
    ax.set_ylabel('X(t2)')
    ax.grid()
    # ax.set_xlim([-100,100])
    # ax.set_ylim([-100,100])


def plot_error_bar(mean_list, std_list, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    # Plot the data
    ax.errorbar(range(len(mean_list)), mean_list, yerr=std_list, fmt='o', label='Accuracy')
    # Add labels and title
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy with Error Bars')

    mean_acc = np.mean(mean_list)
    std_acc = np.std(mean_list)

    # Add a horizontal line for the mean
    ax.axhline(y=mean_acc, color='r', linestyle='--', label='Mean Accuracy')

    # Add a legend
    ax.legend()


def plot_band_powers(time, band_power, channel, trial, t_min, t_max):
    # Initialize frequency bands of interest
    freq_bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']

    # Convert time range to sample indices
    t_start = np.argmin(np.abs(time - t_min))
    t_end = np.argmin(np.abs(time - t_max))

    # Extract band power data for specified channel and trial
    bp = {band: band_power[band][trial, channel, t_start:t_end] for band in freq_bands}

    # Create subplots
    fig, axs = plt.subplots(len(freq_bands), 1, figsize=(10, 20), sharex=True)

    # Loop over frequency bands and plot band power
    for i, band in enumerate(freq_bands):
        axs[i].plot(time[t_start:t_end], bp[band])
        axs[i].set_ylabel(band.capitalize() + ' power')

    # Add x-axis label and title
    axs[-1].set_xlabel('Time (s)')
    fig.suptitle(f'Band powers for channel {channel + 1}, trial {trial + 1}')

    plt.show()


def plot_connectivity(connectivity_matrix, channel_names):
    """
    Create adjacency matrix from connectivity matrix and visualize it using NetworkX.

    Parameters:
    connectivity_matrix (np.ndarray): 2D connectivity matrix with shape (num_channels, num_channels).
    channel_names (list): List of channel names with length num_channels.

    Returns:
    None
    """
    # Create adjacency matrix by thresholding the connectivity matrix
    threshold = np.percentile(connectivity_matrix, 95)  # set threshold to top 5% of values
    adjacency_matrix = (connectivity_matrix > threshold) * 1

    # Create graph object and add nodes with labels
    graph = nx.Graph()
    for i, channel_name in enumerate(channel_names):
        graph.add_node(i, label=channel_name)

    # Add edges between nodes based on adjacency matrix
    for i in range(adjacency_matrix.shape[0]):
        for j in range(i + 1, adjacency_matrix.shape[1]):
            if adjacency_matrix[i, j] == 1:
                graph.add_edge(i, j)

    # Set node positions for visualization
    pos = nx.circular_layout(graph)

    # Draw nodes with labels and edges
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_labels(graph, pos, labels=nx.get_node_attributes(graph, "label"), font_size=12)
    nx.draw_networkx_edges(graph, pos, edge_color="gray")

    # Show plot
    plt.axis("off")
    plt.show()


def plot_single_channel_accuracy(single_channel_mean_accuracy, single_channel_std_accuracy,
                                 save_path, title=None, file_name=None, axs=None, fig=None):
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(14, 8), dpi=300)

    axs.plot(range(single_channel_mean_accuracy.shape[0]), single_channel_mean_accuracy,
             linewidth=2)
    axs.set_xlabel("Feature Vector Index")
    axs.set_ylabel("Accuracy")
    axs.plot(range(single_channel_mean_accuracy.shape[0]), single_channel_mean_accuracy.shape[0] * [0.5],
             color='black')
    if title is not None:
        axs.set_title(title)
    # axs.set_xlim(0, single_channel_mean_accuracy.shape[0])
    axs.set_ylim(0, 1)
    plt.tight_layout()

    if file_name is not None:
        fig.savefig(save_path + file_name)
        fig.savefig(save_path + file_name[:-4] + '.png')

    axs.errorbar(range(single_channel_mean_accuracy.shape[0]),
                 single_channel_mean_accuracy,
                 yerr=single_channel_std_accuracy,
                 fmt='o',
                 label='Accuracy',
                 color='red', capsize=5, elinewidth=1)
    plt.tight_layout()
    if file_name is not None:
        fig.savefig(save_path + file_name[:-4] + '_with_errorbar.svg')
        fig.savefig(save_path + file_name[:-4] + '_with_errorbar.png')
    return axs, fig


def plot_channel_combination_accuracy(mean_accuracy, save_path=None, file_name=None, std_accuracy=None, fig=None,
                                      axs=None,
                                      save_file=True):
    if axs is None:
        fig, axs = plt.subplots(1, 1, figsize=(12, 9), dpi=300)
    axs.plot(range(len(mean_accuracy)),
             mean_accuracy,
             linewidth=2)
    axs.set_xlabel("Feature Vector Index")
    axs.set_ylabel("Accuracy")
    if save_file is True:
        fig.savefig(save_path + 'without_confidence_' + file_name + '.svg')
        fig.savefig(save_path + 'with_confidence_' + file_name + '.png')

    if std_accuracy is not None:
        axs.errorbar(range(len(mean_accuracy)),
                     mean_accuracy,
                     yerr=std_accuracy,
                     fmt='o',
                     label='Accuracy',
                     color='red', capsize=5, elinewidth=1)
        if save_file is True:
            fig.savefig(save_path + 'with_confidence_' + file_name + '.svg')
            fig.savefig(save_path + 'with_confidence_' + file_name + '.png')
    return fig, axs


def plot_loss(losses):
    if isinstance(losses, dict):
        keys = list(losses.keys())
        n_keys = len(keys)

        fig, axs = plt.subplots(n_keys, 1, figsize=(10, n_keys * 3))  # Adjust size as needed

        for i, key in enumerate(keys):
            axs[i].plot(losses[key])
            axs[i].set_title(key)

        plt.tight_layout()
        plt.show()
    else:
        # Plot the optimization process
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.xlabel('Iteration')
        plt.ylabel('Negative Log-Marginal Likelihood')
        plt.title('Optimization of Negative Log-Marginal Likelihood')
        plt.grid(True)
        plt.show()


def plot_result_through_time(time_features, time_idx, mean_accuracy_list_through_time, std_accuracy_list_through_time,
                             number_of_channels_through_time, number_of_channels_through_time_std, save_path, file_name,
                             fontsize,
                             title=None):
    fig, axs = plt.subplots(2, 1, dpi=300, figsize=(10, 10))
    axs[0].errorbar(time_features[time_idx],
                    mean_accuracy_list_through_time,
                    yerr=std_accuracy_list_through_time,
                    fmt='o',
                    label='Accuracy',
                    color='red', capsize=6, elinewidth=2, markersize=9)
    axs[0].plot(time_features[time_idx], mean_accuracy_list_through_time, color='red',
                linewidth=2)

    axs[1].plot(time_features[time_idx], np.array(number_of_channels_through_time) + 1,
                marker='o', label='Feature Vectors', linewidth=2)
    axs[1].errorbar(time_features[time_idx],
                    np.array(number_of_channels_through_time) + 1,
                    yerr=np.array(number_of_channels_through_time_std),
                    fmt='o',
                    label='Feature Vectors',
                    color='red', capsize=6, elinewidth=2, markersize=9)

    axs[1].set_xlabel("Time (sec)", fontsize=fontsize)
    axs[1].set_ylabel("Number of channels", fontsize=fontsize)
    axs[0].set_ylabel("Accuracy", fontsize=fontsize)
    axs[0].plot(time_features[time_idx[np.argmax(mean_accuracy_list_through_time)]],
                np.max(mean_accuracy_list_through_time), marker='o', color='green', markersize=15)

    if title is not None:
        axs[0].set_title(title)

    plt.tight_layout()
    fig.savefig(save_path + file_name + '.svg')
    fig.savefig(save_path + file_name + '.png')


def plotkernelsample(k, ax, xmin=0, xmax=3):
    xx = np.linspace(xmin, xmax, 300)[:, None]
    K = k(xx)
    ax.plot(xx, np.random.multivariate_normal(np.zeros(300), K, 5).T)
    ax.set_title("Samples " + k.__class__.__name__)


def plot_kernel_function(k, x):
    number_of_components = len(k.kernels)
    xx = [np.column_stack([np.zeros_like(x), x])]
    for i in range(1, number_of_components):
        xx.append(np.column_stack([np.ones_like(x) * i, x]))
    xx = np.vstack(xx)

    kernel = k(xx)
    x = x - np.mean(x)
    num_samples = x.shape[0]
    plt.figure()
    fig, axs = plt.subplots(number_of_components, number_of_components)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    for i in range(number_of_components):
        for j in range(number_of_components):
            axs[i, j].plot(x, kernel[int(num_samples / 2) + i * num_samples, j * num_samples:(j + 1) * num_samples])
            axs[i, j].set_title('kernel for channel ' + str(i) + ' and ' + str(j))
    # plt.show()
