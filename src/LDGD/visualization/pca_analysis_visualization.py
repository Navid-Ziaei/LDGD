import matplotlib.pyplot as plt
from scipy.signal import freqz
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
from scipy.stats import mannwhitneyu
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import networkx as nx
from .graph_visualization import *
from scipy.signal import windows, welch


def plot_multitaper_psd(signal, fs, NW=2):
    """
    Plots the power spectral density (PSD) of a signal using the multitaper method.

    Parameters:
    - signal: the time-domain signal
    - fs: the sampling frequency
    - NW: the time half-bandwidth product (choose a value; 2 is common)
    """

    # Number of tapers to use (often 2*NW - 1)
    K = 2 * NW - 1

    # Generate the tapers
    tapers = windows.dpss(signal.shape[-1], NW, K)

    # Compute the multitaper spectrum by averaging over tapers
    psd_mt = np.zeros(signal.shape[-1] // 2 + 1)
    for taper in tapers:
        frequencies, psd = welch(signal * taper, fs, nperseg=signal.shape[-1], noverlap=0, return_onesided=True)
        psd_mt += psd
    psd_mt /= K

    # Plot the PSD
    plt.figure()
    plt.plot(frequencies, 10 * np.log10(psd_mt))
    plt.title('Multitaper PSD of the Signal')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power/Frequency [dB/Hz]')
    plt.grid()
    plt.show()


def plot_continuous_heatmap(time, signal, indicator, channel_names, normalize=True, cmap='winter', save_path=None,
                            file_name='continuous_signal_heatmap'):
    # Find start and end indices for -1 regions
    neg_starts = np.where(np.diff(np.hstack(([0], indicator == -1, [0]))) == 1)[0]
    neg_ends = np.where(np.diff(np.hstack(([0], indicator == -1, [0]))) == -1)[0]

    # Find start and end indices for 1 regions
    pos_starts = np.where(np.diff(np.hstack(([0], indicator == 1, [0]))) == 1)[0]
    pos_ends = np.where(np.diff(np.hstack(([0], indicator == 1, [0]))) == -1)[0]

    if normalize is True:
        min_val = np.quantile(signal, 0.025, axis=-1, keepdims=True)
        max_val = np.quantile(signal, 0.975, axis=-1, keepdims=True)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        normalized_signal[normalized_signal > 1] = 1
        normalized_signal[normalized_signal < 0] = 0
    else:
        normalized_signal = signal

    fig, ax = plt.subplots(figsize=(20, 15), dpi=200)
    plot_eigenvectors_heatmaps(normalized_signal.transpose(), time=time,
                               channel_names=channel_names, fig=fig, ax=ax, cmap=cmap)
    # Add the regions to the plot
    for index, (start, end) in enumerate(zip(neg_starts, neg_ends)):
        if index == 0:
            label = 'black image period'
        else:
            label = None
        ax.axvline(start, linestyle='--', color='black', alpha=0.8, label=label)

    for index, (start, end) in enumerate(zip(pos_starts, pos_ends)):
        if index == 0:
            label = 'white image period'
        else:
            label = None
        ax.axvline(start, color='black', alpha=0.8, label=label)
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + file_name + '.png')


def moving_average(data, M):
    # Creating a simple uniform window
    window = np.ones(M) / M

    # Convolve your data with this window
    smoothed_data = np.convolve(data, window, mode='valid')  # 'valid' ensures output is of size N-M+1
    return smoothed_data


def plot_continuous_signal(continuous_signal, continuous_indicator, channel_names, channel_number, settings,
                           save_path=None, moving_avg_win=0, file_name='continuous_signals'):
    if moving_avg_win > 0:
        signal = moving_average(continuous_signal[channel_number, :], moving_avg_win)
    else:
        signal = continuous_signal[channel_number, :]
    # Create the plot
    fig, ax = plt.subplots(figsize=(60, 8), dpi=200)
    ax.plot(signal, 'k')  # 'k' stands for black color

    # Find start and end indices for -1 regions
    neg_starts = np.where(np.diff(np.hstack(([0], continuous_indicator == -1, [0]))) == 1)[0]
    neg_ends = np.where(np.diff(np.hstack(([0], continuous_indicator == -1, [0]))) == -1)[0]

    # Find start and end indices for 1 regions
    pos_starts = np.where(np.diff(np.hstack(([0], continuous_indicator == 1, [0]))) == 1)[0]
    pos_ends = np.where(np.diff(np.hstack(([0], continuous_indicator == 1, [0]))) == -1)[0]

    # Add the regions to the plot
    for index, (start, end) in enumerate(zip(neg_starts, neg_ends)):
        if index == 0:
            label = 'black image period'
        else:
            label = None
        ax.axvspan(start, end, facecolor='red', alpha=0.5, label=label)

    for index, (start, end) in enumerate(zip(pos_starts, pos_ends)):
        if index == 0:
            label = 'white image period'
        else:
            label = None
        ax.axvspan(start, end, facecolor='green', alpha=0.5, label=label)

    if settings.task == 'imagine':
        # Find start and end indices for 1 regions
        imagination_starts = np.where(np.diff(np.hstack(([0], continuous_indicator == 2, [0]))) == 1)[0]
        imagination_ends = np.where(np.diff(np.hstack(([0], continuous_indicator == 2, [0]))) == -1)[0]

        # Find start and end indices for 1 regions
        choice_starts = np.where(np.diff(np.hstack(([0], continuous_indicator == 3, [0]))) == 1)[0]
        choice_ends = np.where(np.diff(np.hstack(([0], continuous_indicator == 3, [0]))) == -1)[0]

        selection_starts = np.where(np.diff(np.hstack(([0], continuous_indicator == 4, [0]))) == 1)[0]
        selection_ends = np.where(np.diff(np.hstack(([0], continuous_indicator == 4, [0]))) == -1)[0]

        for index, (start, end) in enumerate(zip(imagination_starts, imagination_ends)):
            if index == 0:
                label = 'imagination period'
            else:
                label = None
            ax.axvspan(start, end, facecolor='blue', alpha=0.5, label=label)

        for index, (start, end) in enumerate(zip(choice_starts, choice_ends)):
            if index == 0:
                label = 'choice period'
            else:
                label = None
            ax.axvspan(start, end, facecolor='yellow', alpha=0.5, label=label)

        for index, (start, end) in enumerate(zip(selection_starts, selection_ends)):
            if index == 0:
                label = 'choice onset'
            else:
                label = None
            ax.axvspan(start - 1, end + 1, color='purple', linestyle='--', alpha=1, label=label)

    ax.set_title(channel_names[channel_number])
    ax.legend()
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + f"{file_name}_{channel_names[channel_number]}.png")


def plot_synchronized_avg(x, y, time, time_annotation, channel_names, save_path=None):
    if len(y.shape) > 1:
        for key in y.keys().to_list():
            fig, axs = plt.subplots(1, 3, figsize=(40, 20))
            plot_eigenvectors_heatmaps(np.mean(x[y[key].values == 0], axis=0).transpose(), time=time,
                                       channel_names=channel_names,
                                       title=f'Heatmap of signals {key} 1', fig=fig, ax=axs[0])
            plot_eigenvectors_heatmaps(np.mean(x[y[key].values == 1], axis=0).transpose(), time=time,
                                       channel_names=channel_names,
                                       title=f'Heatmap of signals {key} 2', fig=fig, ax=axs[1])
            plot_eigenvectors_heatmaps(np.mean(x[y[key].values == 0], axis=0).transpose() -
                                       np.mean(x[y[key].values == 1], axis=0).transpose(),
                                       time=time,
                                       channel_names=channel_names,
                                       title=f'Heatmap of difference between {key}', fig=fig, ax=axs[2])
            colors = ['red', 'blue', 'purple', 'black']
            for idx, anot in enumerate(time_annotation.keys()):
                t_event = np.argmin(np.abs(time - time_annotation[anot]))
                axs[0].axvline(t_event, color=colors[idx], linewidth=1.5)
                axs[1].axvline(t_event, color=colors[idx], linewidth=1.5)
                axs[2].axvline(t_event, color=colors[idx], linewidth=1.5, label=key)

            axs[2].legend()
            if save_path is None:
                plt.show()
            else:
                fig.savefig(save_path + f"synchronized_avg_{key}.png")


def plot_time_trace(data, ax=None, label=None):
    times = np.linspace(0, 1, data.shape[0])  # generating 90 values between 0 and 1 for example

    # Create a color map to map time values to colors
    norm = mcolors.Normalize(vmin=times.min(), vmax=times.max())
    if label is None:
        colormap = cm.jet
    elif label == 0:
        colormap = cm.spring
    else:
        colormap = cm.winter

    if ax is None:
        fig, ax = plt.subplots()
    for i in range(1, len(data)):
        ax.plot(data[i - 1:i + 1, 0], data[i - 1:i + 1, 1],
                color=colormap(norm(times[i])), lw=2)

    # Optionally, you can add a colorbar to help indicate the time values
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Time')

    plt.xlabel('Data Dimension 1')
    plt.ylabel('Data Dimension 2')
    plt.title('2D Time Series Data')

    return ax


def plot_eigenvectors_heatmaps_interactive(eigenvector, time, channel_names, title='Heatmap of Eigenvector'):
    # Create hover text for each value
    hovertext = []
    for t, time_value in enumerate(time):
        hovertext.append([f"Time: {time_value}<br>Channel: {channel_names[i]}<br>Value: {eigenvector[t, i]}"
                          for i in range(len(channel_names))])

    fig = go.Figure(data=go.Heatmap(
        z=np.transpose(eigenvector),
        x=time,
        y=channel_names,
        colorscale='Jet',
        hoverongaps=False,
        hoverinfo="text",
        text=hovertext))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Channels")
    fig.show()

    return fig


def plot_eigenvectors_heatmaps(eigenvector, time, channel_names, fig=None, ax=None, title='Heatmap of Eigenvector',
                               cmap='jet'):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(20, 15))
    sns.heatmap(np.transpose(eigenvector), xticklabels=10, yticklabels=10, cmap=cmap, ax=ax)
    ax.set_ylabel('Channels')
    ax.set_xlabel('Time')
    ax.set_title(title)

    inc = int(len(time) / 80)
    # Adjusting xticks and yticks to avoid overlap
    ax.set_yticks(ticks=np.arange(0, len(channel_names), 2), labels=channel_names[0:len(channel_names):2], rotation=0,
                  ha='right')
    ax.set_xticks(ticks=np.arange(0, len(time), inc), labels=np.round(time[0:len(time):inc], 2), rotation=90)

    # Drawing a red line at t=0
    t_zero_index = np.argmin(np.abs(time))
    ax.axvline(t_zero_index, color="red", linewidth=1.5)

    plt.tight_layout()

    return fig, ax


def plot_time_trace_with_array(data, ax=None, color='r'):
    z1_range = np.linspace(np.min(data[:, 0]) * 1.1, np.max(data[:, 0]) * 1.1, num=100)
    z2_range = np.linspace(np.min(data[:, 1]) * 1.1, np.max(data[:, 1]) * 1.1, num=100)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    X = data[:, 0]
    Y = data[:, 1]
    U = np.diff(X)
    V = np.diff(Y)
    ax.plot(X, Y, 'bo-', color=color)
    ax.quiver(X[:-1], Y[:-1], U, V, angles='xy', scale_units='xy', scale=1, color=color)
    ax.set_xlim(np.min(z1_range) * 1.3, np.max(z1_range) * 1.3)
    ax.set_ylim(np.min(z2_range) * 1.3, np.max(z2_range) * 1.3)

    return ax


def plot_time_trace_3d(data, times=None, title='', save_path=None):
    """
    Plots a 3D time series with colors representing time progression.

    Parameters:
    - data: 90x3 numpy array with data values.
    - times: 90x1 numpy array with time values.
    """
    if times is None:
        times = np.linspace(0, 1, data.shape[0])
    # Create a color map to map time values to colors
    norm = mcolors.Normalize(vmin=times.min(), vmax=times.max())
    colormap = cm.jet

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in range(1, len(data)):
        ax.plot(data[i - 1:i + 1, 0], data[i - 1:i + 1, 1], data[i - 1:i + 1, 2],
                color=colormap(norm(times[i])), lw=2)

    # Optionally, you can add a colorbar to help indicate the time values
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label='Time')

    ax.set_xlabel('Data Dimension 1')
    ax.set_ylabel('Data Dimension 2')
    ax.set_zlabel('Data Dimension 3')
    ax.set_title('3D Time Series Data ' + title)
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + f"3D Time Series Data {title}.png")


def plot_clustered_heatmap(data, title, time, channel_names, save_path=None):
    # Compute the distance matrix
    dist_matrix = pdist(data, metric='euclidean')
    dist_matrix = squareform(dist_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(dist_matrix, method='average')

    # Plot dendrogram and heatmap
    ffig, (ax1, ax2) = plt.subplots(1, 2, figsize=(30, 25), dpi=300, gridspec_kw={'width_ratios': [1, 2]})

    # Dendrogram
    dendro = dendrogram(linkage_matrix, orientation='left', ax=ax1, color_threshold=0, above_threshold_color='grey',
                        labels=None)
    ax1.set_yticklabels([])  # Removes labels from dendrogram

    # Rearrange the data according to the clustering
    data_clustered = data[dendro['leaves'], :]

    # Heatmap
    sns.heatmap(data_clustered, cmap='viridis', cbar=True, ax=ax2, xticklabels=np.round(time),
                yticklabels=np.array(channel_names)[dendro['leaves']])
    ax2.set_title(title)

    if save_path is None:
        plt.show()
    else:
        ffig.savefig(save_path + title + '.png')


def plot_dendrogram(data, title, ax, channel_names):
    # Compute the distance matrix
    dist_matrix = pdist(data, metric='euclidean')
    dist_matrix = squareform(dist_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(dist_matrix, method='average')

    # Dendrogram
    dendrogram(linkage_matrix, orientation='left', ax=ax, color_threshold=0, above_threshold_color='grey',
               labels=channel_names)
    ax.set_title(title)


def plot_channel_graph(data, title, channel_names, top_n=5, save_path=None):
    # Compute the distance matrix
    dist_matrix = pdist(data, metric='euclidean')
    dist_matrix = squareform(dist_matrix)

    # Convert distance matrix to similarity matrix (one way is to use inverse)
    similarity_matrix = 1 / (1 + dist_matrix)

    # Create graph
    G = nx.from_numpy_array(similarity_matrix)

    # For each node, sort its edges by weight and keep the top_n
    top_edges = []
    for node in G.nodes():
        edges_sorted = sorted([(n, e) for n, e in G[node].items() if n != node], key=lambda x: x[1]['weight'],
                              reverse=True)[:top_n]
        top_edges.extend([(node, edge[0]) for edge in edges_sorted])

    # Create a new graph with only the top_n edges for each node
    H = nx.Graph()
    for edge in top_edges:
        H.add_edge(edge[0], edge[1], weight=G[edge[0]][edge[1]]['weight'])

    # Add channel names as labels
    label_mapping = dict(zip(H.nodes(), channel_names))
    H = nx.relabel_nodes(H, label_mapping)

    # Draw the graph
    fig = plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(H)  # positions for all nodes
    nx.draw_networkx_nodes(H, pos, node_size=500)
    nx.draw_networkx_edges(H, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(H, pos, font_size=10, font_family="sans-serif")
    plt.title(title)

    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + title + "graph.png")


def plot_Scree(time, explained_variance, save_path=None):
    cumulative_variance = np.cumsum(explained_variance, axis=1)
    fig = plt.figure()
    plt.plot(cumulative_variance.transpose())
    plt.vlines(np.argmin(np.abs(cumulative_variance[0] - 0.9)),
               ymin=np.min(cumulative_variance),
               ymax=np.max(cumulative_variance))
    plt.xlabel('Number of Components')
    plt.ylabel(f'Cumulative Explained Variance ')
    plt.title(f'Scree Plot for all timepoints (knee point = {np.argmin(np.abs(cumulative_variance[0] - 0.9))})')
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + "scree_plot.png")

    knee_points = [np.argmin(np.abs(cumulative_variance[i] - 0.9)) for i in range(cumulative_variance.shape[0])]
    fig = plt.figure()
    plt.plot(time, knee_points)
    plt.xlabel('Time')
    plt.ylabel(f'Knee point (90% explained variance)')
    plt.title(f'Temporal Dynamics of Explained Variance')

    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + "scree_plot_knee_point_throgh_time.png")


def plot_pca_component_dynamic(time, PCs_white, PCs_black, save_path=None):
    for i in range(7):
        fig = plt.figure(figsize=(8, 6))
        plt.plot(time, [np.mean(pc[:, i]) for pc in PCs_black], label='Black - 1st PC', color='green')
        plt.plot(time, [np.mean(pc[:, i]) for pc in PCs_white], label='White - 1st PC', color='red')
        plt.xlabel('Time (ms)')
        plt.ylabel('Principal Component Value')
        plt.legend()
        plt.title(f"Time Evolution of {i}'th Principal Component")
        plt.tight_layout()

        if save_path is None:
            plt.show()
        else:
            fig.savefig(save_path + f"component{i}_dynamic.png")


def plot_clusers(PCs_black_array, PCs_white_array, time, channel_names, save_path=None):
    # For Black Stimulus
    plot_clustered_heatmap(np.mean(PCs_black_array, axis=1).transpose(),
                           'Clustered Heatmap for Black Stimulus', time, channel_names, save_path=save_path)

    # For White Stimulus
    plot_clustered_heatmap(np.mean(PCs_white_array, axis=1).transpose(),
                           'Clustered Heatmap for White Stimulus', time, channel_names, save_path=save_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(25, 12), dpi=300)

    # For Black Stimulus
    plot_dendrogram(np.mean(PCs_black_array, axis=1).transpose(),
                    'Dendrogram for Black Stimulus', ax1, channel_names)
    # For White Stimulus
    plot_dendrogram(np.mean(PCs_white_array, axis=1).transpose(),
                    'Dendrogram for White Stimulus', ax2, channel_names)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + f"dendrogram.png")

    # Plot for Black Stimulus
    plot_channel_graph(np.mean(PCs_black_array, axis=1).transpose(),
                       'Channel Graph for Black Stimulus', channel_names, save_path=save_path)

    # Plot for White Stimulus
    plot_channel_graph(np.mean(PCs_white_array, axis=1).transpose(),
                       'Channel Graph for White Stimulus', channel_names, save_path=save_path)


def plot_difference_pca_component(PCs_black_array, PCs_white_array, time, top_n=7, save_path=None):
    fig = plt.figure(figsize=(12, 8))
    difference = np.mean(PCs_black_array, axis=1).transpose() - np.mean(PCs_white_array, axis=1).transpose()
    plt.plot(time, difference[:top_n].transpose())
    plt.title('Difference between Black and White Stimulus PCs')
    plt.legend([str(i) for i in range(top_n)])
    fig.savefig("component_difference_dynamic.png")
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + "component_difference_dynamic.png")


def plot_boxplot_pca_components(pcs_black_array, pcs_white_array, time, save_path=None):
    # Compute the mean across trials (axis=1)
    # Compute the mean across trials (axis=1)
    mean_black = np.mean(pcs_black_array, axis=1)
    mean_white = np.mean(pcs_white_array, axis=1)
    n_timepoints = mean_black.shape[0]
    data_long = []
    for i in range(n_timepoints):
        for j in range(mean_black.shape[1]):
            data_long.append([np.round(time[i]), 'Black', mean_black[i, j]])
            data_long.append([np.round(time[i]), 'White', mean_white[i, j]])

    df = pd.DataFrame(data_long, columns=['Timepoint', 'Stimulus', 'Value'])

    # Box Plot
    fig = plt.figure(figsize=(60, 6))
    sns.boxplot(x='Timepoint', y='Value', hue='Stimulus', data=df)
    plt.title('Distribution of Principal Component Values across Time')
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + "boxplot.png")

    # Violin Plot
    fig = plt.figure(figsize=(60, 10))
    sns.violinplot(x='Timepoint', y='Value', hue='Stimulus', data=df, split=True)
    plt.title('Distribution of Principal Component Values across Time')
    plt.xticks(rotation=90)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + "violinplot.png")


def plot_pvalues_over_time(pcs_black_array, pcs_white_array, time, save_path=None):
    n_timepoints = pcs_black_array.shape[0]
    p_values = []

    for i in range(n_timepoints):
        # Flatten over channels to get a 1D array of PCA values for each stimulus at a particular time point
        pca_vals_black = pcs_black_array[i].flatten()
        pca_vals_white = pcs_white_array[i].flatten()

        # Mann-Whitney U test
        _, p = mannwhitneyu(pca_vals_black, pca_vals_white, alternative='two-sided')
        p_values.append(p)

    # Plotting p-values over time
    fig = plt.figure(figsize=(20, 6))
    plt.plot(time, p_values, '-o', color='blue', markersize=3)
    plt.axhline(y=0.05, color='red', linestyle='--')  # significance level line
    plt.xlabel('Time (ms)')
    plt.ylabel('p-value')
    plt.title('P-values over Time for PCA values: Black vs White')
    plt.yscale('log')  # use logarithmic scale for clarity
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + "pvalues_plot.png")

    return p_values


def plot_compare_explained_variance(time, eigenvalues1, eigenvalues2, title='Eigen Value', fig=None, axs=None,
                                    save_path=None):
    if axs is None:
        fig, axs = plt.subplots(3, figsize=(9, 16))
    axs[0].plot(time, np.stack(eigenvalues1))
    axs[1].plot(time, np.stack(eigenvalues2))
    axs[2].plot(time, np.stack(eigenvalues2) - np.stack(eigenvalues1))
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Eigen Value")
    axs[0].set_title(f"{title} s through time for black square")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Eigen Value")
    axs[1].set_title(f"{title} s through time for white square")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Eigen Value Difference")
    axs[2].set_title(f"{title} Black and White difference through time")
    plt.legend([f"eigen value {i}" for i in range(np.stack(eigenvalues1).shape[0])])
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + title + '_difference_dynamic.png')

    return axs, fig


def angle_between_vectors(v1, v2):
    cosine_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle_rad = np.arccos(np.clip(cosine_sim, -1, 1))  # clip to handle potential numerical errors
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def plot_eigenvector_heatmap_2class(eigenvectors1, eigenvectors2, component_idx, time, time_annotation, channel_names,
                                    title_suffix='',
                                    save_path=None):
    eigenvector1 = np.stack([ev[component_idx, :] for ev in eigenvectors1])
    eigenvector2 = np.stack([ev[component_idx, :] for ev in eigenvectors2])
    fig, axs = plt.subplots(1, 3, figsize=(60, 15))
    plot_eigenvectors_heatmaps(eigenvector1, time=time, channel_names=channel_names, ax=axs[0], fig=fig,
                               title=f'Heatmap of Eigenvector for eigen vector {component_idx} (Black)' + title_suffix)
    plot_eigenvectors_heatmaps(eigenvector2, time=time, channel_names=channel_names, ax=axs[1], fig=fig,
                               title=f'Heatmap of Eigenvector for eigen vector {component_idx} (White)' + title_suffix)
    plot_eigenvectors_heatmaps(np.abs(eigenvector1 - eigenvector2), time=time,
                               channel_names=channel_names, ax=axs[2], fig=fig,
                               title=f'Heatmap of Eigenvector for eigen vector {component_idx} (Difference)' +
                                     title_suffix)

    colors = ['red', 'blue', 'purple', 'black']
    for idx, key in enumerate(time_annotation.keys()):
        t_event = np.argmin(np.abs(time - time_annotation[key]))
        axs[0].axvline(t_event, color=colors[idx], linewidth=1.5)
        axs[1].axvline(t_event, color=colors[idx], linewidth=1.5)
        axs[2].axvline(t_event, color=colors[idx], linewidth=1.5, label=key)

    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + f"Eigenvector_dynamic_heatmap_component{component_idx + 1}_{title_suffix}.png")

    # plot_eigenvectors_heatmaps_interactive(eigenvector1, time, channel_names, title='Heatmap of Eigenvector Black')
    # plot_eigenvectors_heatmaps_interactive(eigenvector2, time, channel_names, title='Heatmap of Eigenvector White')


def plot_covmat_heatmap_2class(pca_results1, pca_results2, time, time_idx, channel_names, save_path=None, key=''):
    fig, ax = plt.subplots(1, 2, figsize=(33, 15))
    cov1 = np.transpose(pca_results1[time_idx].get_covariance())
    cov2 = np.transpose(pca_results2[time_idx].get_covariance())
    sns.heatmap(cov1, xticklabels=10, yticklabels=10, cmap='jet', ax=ax[0])
    sns.heatmap(cov2, xticklabels=10, yticklabels=10, cmap='jet', ax=ax[1])

    ax[0].set_ylabel('Channels')
    ax[0].set_xlabel('Channels')
    ax[0].set_title(f"Covariance for Black stimiuli for t = {time[time_idx]}", fontsize=20)
    ax[0].set_yticks(ticks=np.arange(0, len(channel_names), 2), labels=channel_names[0:len(channel_names):2],
                     rotation=0, ha='right')
    ax[0].set_xticks(ticks=np.arange(0, len(channel_names), 2), labels=channel_names[0:len(channel_names):2],
                     rotation=90)

    ax[1].set_ylabel('Channels')
    ax[1].set_xlabel('Channels')
    ax[1].set_title(f"Covariance for White stimiuli for t = {time[time_idx]}", fontsize=20)
    ax[1].set_yticks(ticks=np.arange(0, len(channel_names), 2), labels=channel_names[0:len(channel_names):2],
                     rotation=0, ha='right')
    ax[1].set_xticks(ticks=np.arange(0, len(channel_names), 2), labels=channel_names[0:len(channel_names):2],
                     rotation=90)

    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + f"Covariance_time{time[time_idx]}_{key}.png")


def plot_freq_response_fir(taps, fs):
    # Calculate the frequency response
    w, h = freqz(taps, worN=8000)

    # Convert the frequencies to Hz
    frequencies = np.linspace(0, fs // 2, len(w))

    # Plot the magnitude response (in dB)
    plt.figure()
    plt.plot(frequencies, 20 * np.log10(np.abs(h)))
    plt.title('FIR Bandpass Filter Frequency Response')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Gain [dB]')
    plt.grid()
    plt.show()


def plot_fft(signal, fs):
    """
    Plots the FFT of a signal.

    Parameters:
    - signal: the time-domain signal
    - fs: the sampling frequency
    """
    # Compute the FFT
    N = len(signal)
    fft_values = np.fft.rfft(signal)

    # Compute the frequencies for the positive half of the spectrum
    frequencies = np.fft.rfftfreq(N, 1 / fs)

    # Plot the magnitude spectrum
    plt.figure()
    plt.plot(frequencies, 20 * np.log10(np.abs(fft_values)))
    plt.title('FFT of the Signal')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid()
    plt.show()


def plot_signals_in_a_channel(x, y, selected_channel, time, channel_names, offset_inc=0.05, save_path=None, key=''):
    fig = plt.figure(figsize=(12, 8))
    offset = 0
    for i in range(0, x.shape[0]):
        if y[i] == 0:
            color = 'red'
        else:
            color = 'green'
        plt.plot(time,
                 x[i, selected_channel, :] +
                 offset, color=color, alpha=0.8
                 )
        offset = offset + offset_inc
    plt.title(channel_names[selected_channel])
    plt.xlabel("Time")
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + f"signals_{selected_channel}.png")

    fig = plt.figure(figsize=(12, 8))
    offset = 0
    for i in range(0, x.shape[0]):
        plt.plot(time,
                 x[i, selected_channel, :] +
                 offset, alpha=0.8
                 )
        offset = offset + offset_inc
    plt.title(channel_names[selected_channel])
    plt.xlabel("Time")
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + f"signals_{selected_channel}_colorfull_{key}.png")


def plot_angle_between_vectors(eigenvectors1, eigenvectors2, componenet_idx, time, fig=None, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, figsize=(10, 8))
    eigenvector1 = np.stack([ev[componenet_idx, :] for ev in eigenvectors1])
    eigenvector2 = np.stack([ev[componenet_idx, :] for ev in eigenvectors2])

    angles = []
    square_dist = []

    for j in range(eigenvector1.shape[0]):
        angles.append(angle_between_vectors(eigenvector1[j], eigenvector2[j]))
        square_dist.append(np.linalg.norm(eigenvector1[j] - eigenvector2[j]))

    axs.plot(time, angles)
    axs.set_title(f"Phase difference between black and white eigen vector {componenet_idx} throgh time")
    axs.set_xlabel("time")
    axs.set_ylabel("Angle (degree)")


def create_animation_heatmap_topo(eigenvectors, componenet_idx, channel_names, title='black', save_path=''):
    eigenvector = np.stack([ev[componenet_idx, :] for ev in eigenvectors])
    # Create a figure and axis for the animation
    fig, ax = plt.subplots(1, 1, figsize=(30, 20))

    # Create an initial colorbar (will be updated during the animation)
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=ax, orientation='vertical')
    cbar.set_label('Activation Level', size=16)

    # Update function for the animation
    def update(frame):
        ax.clear()
        plot_brain_heatmap(eigenvector[frame], channel_names, fig=fig, ax=ax, colorbar=cbar)

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(eigenvector), repeat=True)

    # To save the animation as a video file
    ani.save(save_path + f'heatmap_animation{componenet_idx}_{title}.mp4', writer='ffmpeg', fps=1)


def plot_pca_trajectory(z_t_pca, y, win_len, save_path=None):
    y_test_reshaped = np.tile(y, win_len)
    time_series_pca = np.stack(z_t_pca)

    single_patient_data1_pca = np.mean(time_series_pca[:, y_test_reshaped == 0, :], axis=1)
    single_patient_data2_pca = np.mean(time_series_pca[:, y_test_reshaped == 1, :], axis=1)

    fig, axs = plt.subplots(1, 1, figsize=(12, 12))
    plot_time_trace(single_patient_data1_pca, ax=axs, label=0)
    plot_time_trace(single_patient_data2_pca, ax=axs, label=1)
    plt.tight_layout()
    plt.title("PCA components trajectory through time ")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + f"both.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_time_trace(single_patient_data1_pca, ax=axs[0])
    plot_time_trace(single_patient_data2_pca, ax=axs[1])
    # Get the x and y limits from both axes
    xlims1 = axs[0].get_xlim()
    ylims1 = axs[0].get_ylim()
    xlims2 = axs[1].get_xlim()
    ylims2 = axs[1].get_ylim()

    # Determine the common maximum range for x and y
    common_xlims = (min(xlims1[0], xlims2[0]), max(xlims1[1], xlims2[1]))
    common_ylims = (min(ylims1[0], ylims2[0]), max(ylims1[1], ylims2[1]))

    # Apply these common limits to both axes
    axs[0].set_xlim(common_xlims)
    axs[0].set_ylim(common_ylims)
    axs[0].set_xlabel("PCA component 1")
    axs[0].set_ylabel("PCA component 2")
    axs[0].set_title("The mean of all black square trials")
    axs[1].set_xlim(common_xlims)
    axs[1].set_ylim(common_ylims)
    axs[1].set_xlabel("PCA component 1")
    axs[1].set_ylabel("PCA component 2")
    axs[1].set_title("The mean of all black white trials")
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + f"mean_trajectory_subplot.png")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    plot_time_trace(time_series_pca[:, y_test_reshaped == 0, :], ax=axs[0])
    plot_time_trace(time_series_pca[:, y_test_reshaped == 1, :], ax=axs[1])
    # Get the x and y limits from both axes
    xlims1 = axs[0].get_xlim()
    ylims1 = axs[0].get_ylim()
    xlims2 = axs[1].get_xlim()
    ylims2 = axs[1].get_ylim()

    # Determine the common maximum range for x and y
    common_xlims = (min(xlims1[0], xlims2[0]), max(xlims1[1], xlims2[1]))
    common_ylims = (min(ylims1[0], ylims2[0]), max(ylims1[1], ylims2[1]))

    # Apply these common limits to both axes
    axs[0].set_xlim(common_xlims)
    axs[0].set_ylim(common_ylims)
    axs[1].set_xlim(common_xlims)
    axs[1].set_ylim(common_ylims)
    plt.tight_layout()
    if save_path is None:
        plt.show()
    else:
        fig.savefig(save_path + f"all_trajectory_subplot.png")

    plot_time_trace_3d(single_patient_data1_pca, save_path=save_path, title='black')
    plot_time_trace_3d(single_patient_data2_pca, save_path=save_path, title='white')
