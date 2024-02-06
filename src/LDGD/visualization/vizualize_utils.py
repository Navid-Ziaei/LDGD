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


class DataVisualizer:
    def __init__(self, data, label_name=None):
        self.fs = data.fs
        self.time = data.time
        if isinstance(data.label, pd.DataFrame):
            self.label = data.label.values.astype(int)
        else:
            self.label = data.label.astype(int)
        self.channel_name = data.channel_name
        if label_name is None:
            self.label_name = [str(i) for i in range(len(np.unique(self.label)))]
        else:
            self.label_name = label_name

    def plot_single_channel_data(self, data, trial_idx, channel_idx, t_min=None, t_max=None, ax=None, alpha=1,
                                 color=None):
        """

        :param data:
        :param trial_idx:
        :param channel_idx:
        :param t_min:
        :param t_max:
        :param ax:
        :return:
        """
        # Convert time interval to sample indices
        start_idx = np.argmin(np.abs(self.time - t_min)) if t_min is not None else 0
        end_idx = np.argmin(np.abs(self.time - t_max)) if t_max is not None else len(self.time)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.plot(self.time[start_idx:end_idx], data[trial_idx, channel_idx, start_idx:end_idx], alpha=alpha, color=color)
        ax.set_xlabel("Time (second)")
        ax.set_ylabel("Amplitude")
        ax.set_title(self.channel_name[channel_idx] + ' Label = ' + self.label_name[self.label[trial_idx]])

        return ax

    def plot_sync_avg_with_ci(self, data, channel_idx, t_min=None, t_max=None, ci=0.95, ax=None):
        """
        Plot synchronous average of trials with confidence interval.

        Parameters
        ----------
        :param data : numpy.ndarray
            Data array of shape (num_trials, num_channels, num_samples).
        :param channel_idx : int
            Index of the channel to plot.
        :param ci : float, optional
            Confidence interval, default is 0.95.
        :param ax: matplotlib.axes._subplots.AxesSubplot
            Matplotlib ax, default is None
        :param t_min:
        :param t_max:
        Returns
        -------
        ax : matplotlib.axes._subplots.AxesSubplot
            The axis object containing the plot.

        """
        start_idx = np.argmin(np.abs(self.time - t_min)) if t_min is not None else 0
        end_idx = np.argmin(np.abs(self.time - t_max)) if t_max is not None else len(self.time)
        # Get the data for the specified channel
        channel_data = data[:, channel_idx, start_idx:end_idx]

        # Calculate the synchronous average
        sync_avg = np.mean(channel_data, axis=0)

        # Calculate the standard error of the mean
        sem = stats.sem(channel_data, axis=0)
        # Calculate the confidence interval
        h = sem * stats.t.ppf((1 + ci) / 2, len(channel_data) - 1)

        ci_low = sync_avg - h
        ci_high = sync_avg + h

        # Create the plot
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(self.time[start_idx:end_idx], sync_avg, color='black')
        ax.fill_between(self.time[start_idx:end_idx], ci_low, ci_high, alpha=0.2)

        # Set the axis labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Synchronous Average (Channel {channel_idx} :' + self.channel_name[channel_idx] + ' )')

        return ax

    def plot_power_spectrum(self, data, channel_idx, trial_idx, t_min, t_max, ax=None, enable_plot=False, use_log=True):
        """
        Compute power spectrum for a given channel and trial index, and time interval between t_min and t_max.

        Parameters:
            data (ndarray): EEG data array of shape (num_trials, num_channels, num_samples).
            channel_idx (int): Index of the channel of interest.
            trial_idx (int): Index of the trial of interest.
            t_min (float): Start time in seconds of the interval of interest.
            t_max (float): End time in seconds of the interval of interest.
            ax (matplotlib axe):

        Returns:
            f (ndarray): Frequency vector.
            psd (ndarray): Power spectral density for the selected channel and trial.
        """
        # Get the index range for the time interval
        t_start = np.argmin(np.abs(self.time - t_min))
        t_end = np.argmin(np.abs(self.time - t_max))

        # Get the EEG data for the selected channel and trial within the time interval
        eeg_data = data[trial_idx, channel_idx, t_start:t_end]

        # Compute the power spectral density using the Welch method with a Hann window
        f, psd = signal.welch(eeg_data, fs=self.fs, window='hann', nperseg=1024, noverlap=512)

        if enable_plot is True:
            if ax is None:
                fig, ax = plt.subplots(1, 1)
            if use_log is True:
                ax.plot(f, np.log(abs(psd)))
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Log(Power spectral density)")
            else:
                ax.plot(f, abs(psd))
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Log(Power spectral density)")
            ax.set_title(self.channel_name[channel_idx] + ' Label = ' + self.label_name[self.label[trial_idx]])

        return f, psd

    def plot_average_power_spectrum(self, data, channel_idx, t_min, t_max, alpha=0.05, ax=None):
        """
        Compute and plot the average power spectrum over multiple trials with a confidence interval.

        Parameters:
            data (ndarray): EEG data array of shape (num_trials, num_channels, num_samples).
            channel_idx (int): Index of the channel of interest.
            t_min (float): Start time in seconds of the interval of interest.
            t_max (float): End time in seconds of the interval of interest.
            alpha (float): Significance level for the confidence interval. Default is 0.05.
            ax (matplotlib axe):
        """
        # Compute the power spectral density for each trial
        psd_all_trials = []
        for trial_idx in range(data.shape[0]):
            _, psd = self.plot_power_spectrum(data, channel_idx, trial_idx, t_min, t_max)
            psd_all_trials.append(psd)

        # Compute the average power spectral density and confidence interval
        psd_all_trials = np.array(psd_all_trials)
        psd_mean = np.mean(psd_all_trials, axis=0)
        psd_std = np.std(psd_all_trials, axis=0, ddof=1)
        t_value = stats.t.ppf(1 - alpha / 2, data.shape[0] - 1)
        ci = t_value * psd_std / np.sqrt(data.shape[0])

        # Plot the average power spectrum with confidence interval
        f = _  # reusing frequency vector from previous function call
        ax.plot(f, psd_mean, color='black')
        ax.fill_between(f, psd_mean - ci, psd_mean + ci, color='gray', alpha=0.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power spectral density')

        return ax


def plot_results_gplvm(X, std, labels, losses, inverse_length_scale, latent_dim, largest=True, save_path=None,
                       file_name='gplvm_result', show_errorbars=True):
    values, indices = torch.topk(torch.tensor(inverse_length_scale), k=2, largest=largest)
    l1 = indices.numpy().flatten()[0]
    l2 = indices.numpy().flatten()[1]

    colors = ['r', 'b', 'g']

    plt.figure(figsize=(20, 8))

    # 2D Latent Subspace Plot
    ax1 = plt.subplot(131)
    # ax1.set_title('2D Latent Subspace Corresponding to 3 Phase Oilflow', fontsize=32)
    ax1.set_xlabel(f'Latent Dimension {l1+1}', fontsize=32)
    ax1.set_ylabel(f'Latent Dimension {l2+1}', fontsize=32)
    ax1.tick_params(axis='both', labelsize=28)
    # Applying consistent spines format
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1)
    ax1.spines['bottom'].set_linewidth(1)
    # Applying consistent grid format
    for i, label in enumerate(np.unique(labels)):
        X_i = X[labels == label]
        scale_i = std[labels == label]
        ax1.scatter(X_i[:, l1], X_i[:, l2], c=colors[i], label=f'Class {label+1}', s=40, edgecolor=colors[i], alpha=1)
        if show_errorbars is True:
            ax1.errorbar(X_i[:, l1], X_i[:, l2], xerr=scale_i[:, l1], yerr=scale_i[:, l2], fmt='none', ecolor=colors[i],
                         alpha=0.5, label=f'Confidence {label+1}')
    ax1.legend(fontsize=24)

    # ARD Coefficients Plot
    ax2 = plt.subplot(132)
    ax2.bar(np.arange(latent_dim), height=inverse_length_scale.flatten())
    ax2.set_xlabel("ARD Coefficients", fontsize=32)
    ax2.set_ylabel("Value", fontsize=32)
    # ax2.set_title('Inverse Lengthscale with SE-ARD Kernel', fontsize=32)
    ax2.tick_params(axis='both', labelsize=28)
    # Applying consistent spines format
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1)
    ax2.spines['bottom'].set_linewidth(1)

    # Neg. ELBO Loss Plot
    ax3 = plt.subplot(133)
    ax3.plot(losses[10:], label='batch_size=100')
    ax3.set_xlabel("Iteration", fontsize=32)
    ax3.set_ylabel("ELBO Loss", fontsize=32)
    # ax3.set_title('Neg. ELBO Loss', fontsize=32)
    ax3.tick_params(axis='both', labelsize=28)
    ax3.legend(fontsize=24)
    # Applying consistent spines format
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.spines['left'].set_linewidth(1)
    ax3.spines['bottom'].set_linewidth(1)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}{file_name}.png')
        plt.savefig(f'{save_path}{file_name}.svg')
    else:
        plt.show()


def plot_heatmap(x, labels, model, alpha, x_std=None, cmap='winter', range_scale=1.2, save_path=None,
                 file_name='latent_heatmap', inducing_points=None, ind_point=0):
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
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + file_name + '.png')
        fig.savefig(save_path + file_name + '.svg')

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
