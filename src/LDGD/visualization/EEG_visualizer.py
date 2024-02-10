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
