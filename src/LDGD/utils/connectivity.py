import numpy as np
from scipy.signal import coherence
from scipy.signal import hilbert
from scipy.stats import pearsonr
from sklearn.metrics import mutual_info_score


class ConnectivityFeatures:
    """
    A class for computing connectivity features from multichannel data.

    Args:
        data (ndarray): The multichannel data of shape (n_trials, n_channels, n_samples).
        fs (float): The sampling frequency of the data.
        time (ndarray): The time vector of shape (n_samples,) representing the time points.
        channel_names (list): A list of channel names corresponding to the data channels.

    Attributes:
        data (ndarray): The multichannel data.
        fs (float): The sampling frequency of the data.
        time (ndarray): The time vector.
        channel_names (list): The list of channel names.
        num_channel (int): The number of channels.

    """

    def __init__(self, data, fs, time, channel_names):
        self.data = data
        self.fs = fs
        self.time = time
        self.channel_names = channel_names
        self.num_channel = len(channel_names)

    def coherence(self, channel1, channel2, t_min, t_max):
        """Calculate coherence between two channels within a specified time interval.

        Args:
            channel1 (int): Index of the first channel.
            channel2 (int): Index of the second channel.
            t_min (float): Start time of the interval of interest in seconds.
            t_max (float): End time of the interval of interest in seconds.

        Returns:
            f (ndarray): The frequencies.
            cxy_mean (ndarray): The average coherence between the channels across frequencies.

        """
        idx_min = np.argmin(np.abs(self.time - t_min))
        idx_max = np.argmin(np.abs(self.time - t_max))
        f, cxy = coherence(
            self.data[:, channel1, idx_min:idx_max],
            self.data[:, channel2, idx_min:idx_max],
            fs=self.fs
        )
        cxy_mean = cxy.mean(axis=0)
        return f, cxy_mean

    def coherence_matrix(self, t_min, t_max):
        """Calculate coherence between all pairs of channels within a specified time interval.

        Args:
            t_min (float): Start time of the interval of interest in seconds.
            t_max (float): End time of the interval of interest in seconds.

        Returns:
            coherence_mat (ndarray): The coherence matrix of shape (num_channel, num_channel).

        """
        coherence_mat = np.zeros((self.num_channel, self.num_channel))
        for i in range(self.num_channel):
            for j in range(i + 1, self.num_channel):
                f, coherence_val = self.coherence(i, j, t_min, t_max)
                coherence_mat[i, j] = coherence_val
                coherence_mat[j, i] = coherence_val
        return coherence_mat

    def phase_sync(self, channel1, channel2, t_min, t_max):
        """Calculate phase synchronization between two channels within a specified time interval.

        Args:
            channel1 (int): Index of the first channel.
            channel2 (int): Index of the second channel.
            t_min (float): Start time of the interval of interest in seconds.
            t_max (float): End time of the interval of interest in seconds.

        Returns:
            ps (ndarray): The phase synchronization values.

        """
        idx_min = np.argmin(np.abs(self.time - t_min))
        idx_max = np.argmin(np.abs(self.time - t_max))
        phase1 = np.angle(hilbert(self.data[:, channel1, idx_min:idx_max]))
        phase2 = np.angle(hilbert(self.data[:, channel2, idx_min:idx_max]))
        ps = np.abs(np.mean(np.exp(1j * (phase1 - phase2)), axis=0))
        return ps

    def phase_sync_matrix(self, t_min, t_max):
        """Calculate phase synchronization between all pairs of channels within a specified time interval.

        Args:
            t_min (float): Start time of the interval of interest in seconds.
            t_max (float): End time of the interval of interest in seconds.

        Returns:
            ps_mat (ndarray): The phase synchronization matrix of shape (num_channel, num_channel).

        """
        ps_mat = np.zeros((self.num_channel, self.num_channel))
        for i in range(self.num_channel):
            for j in range(i + 1, self.num_channel):
                ps = self.phase_sync(i, j, t_min, t_max)
                ps_mat[i, j] = ps
                ps_mat[j, i] = ps
        return ps_mat

    def granger_causality(self, channel1, channel2, max_order=10):
        """Calculate Granger causality between two channels.

        Args:
            channel1 (str): Name of the first channel.
            channel2 (str): Name of the second channel.
            max_order (int, optional): The maximum order of the autoregressive model. Defaults to 10.

        Returns:
            gc (ndarray): The Granger causality values.

        """
        idx1 = self.channel_names.index(channel1)
        idx2 = self.channel_names.index(channel2)
        x = self.data[:, idx1, :]
        y = self.data[:, idx2, :]
        n_trials = x.shape[0]
        n_samples = x.shape[1]
        gc = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            data = np.vstack([x[:, :i+1], y[:, :i+1]])
            gc_xy = 0
            gc_yx = 0
            for j in range(n_trials):
                X = data[:n_trials, :i+1]
                Y = data[n_trials:, :i+1]
                model_xy = np.var(X)
                results_xy = model_xy.fit(maxlags=max_order)
                model_yx = np.var(Y)
                results_yx = model_yx.fit(maxlags=max_order)
                gc_xy += results_xy.params[-1, n_trials:]
                gc_yx += results_yx.params[-1, :n_trials]
            gc_xy /= n_trials
            gc_yx /= n_trials
            gc[i, idx1] = np.sqrt(np.mean(gc_xy ** 2))
            gc[i, idx2] = np.sqrt(np.mean(gc_yx ** 2))
        return gc

    def mutual_info(self, channel1, channel2, t_min, t_max):
        """Calculate mutual information between two channels within a specified time interval.

        Args:
            channel1 (int): Index of the first channel.
            channel2 (int): Index of the second channel.
            t_min (float): Start time of the interval of interest in seconds.
            t_max (float): End time of the interval of interest in seconds.

        Returns:
            mi (float): The mutual information value.

        """
        idx_min = np.argmin(np.abs(self.time - t_min))
        idx_max = np.argmin(np.abs(self.time - t_max))
        x1 = self.data[:, channel1, idx_min:idx_max]
        x2 = self.data[:, channel2, idx_min:idx_max]
        mi = mutual_info_score(x1.flatten(), x2.flatten())
        return mi

    def mutual_info_matrix(self, t_min, t_max):
        """Calculate mutual information for all pairs of channels within a specified time interval.

        Args:
            t_min (float): Start time of the interval of interest in seconds.
            t_max (float): End time of the interval of interest in seconds.

        Returns:
            mi_matrix (ndarray): The mutual information matrix of shape (num_channel, num_channel).

        """
        mi_matrix = np.zeros((self.num_channel, self.num_channel))
        for i in range(self.num_channel):
            for j in range(i + 1, self.num_channel):
                mi = self.mutual_info(i, j, t_min, t_max)
                mi_matrix[i, j] = mi
                mi_matrix[j, i] = mi
        return mi_matrix

    def cross_corr(self, channel1, channel2, t_min, t_max):
        """Calculate cross-correlation between two channels within a specified time interval.

        Args:
            channel1 (int): Index of the first channel.
            channel2 (int): Index of the second channel.
            t_min (float): Start time of the interval of interest in seconds.
            t_max (float): End time of the interval of interest in seconds.

        Returns:
            xcorr (ndarray): The cross-correlation values.

        """
        idx_min = np.argmin(np.abs(self.time - t_min))
        idx_max = np.argmin(np.abs(self.time - t_max))
        x1 = self.data[:, channel1, idx_min:idx_max]
        x2 = self.data[:, channel2, idx_min:idx_max]
        xcorr = np.mean(np.correlate(x1, x2, mode='same'), axis=0)
        return xcorr

    def cross_corr_matrix(self, t_min, t_max):
        """
        Calculates the cross-correlation between all pairs of channels within a specific time interval.

        Args:
            t_min (float): The start time of the interval of interest in seconds.
            t_max (float): The end time of the interval of interest in seconds.

        Returns:
            cross_corr (ndarray): The cross-correlation matrix of shape (num_channel, num_channel) for the given time interval.

        """
        # Calculate the cross-correlation for all pairs of channels
        cross_corr = np.zeros((self.num_channel, self.num_channel))
        for i in range(self.num_channel):
            for j in range(i + 1, self.num_channel):
                cr = self.cross_corr(i, j, t_min, t_max)
                cross_corr[i, j] = cr
                cross_corr[j, i] = cr

        return cross_corr

    def phase_amplitude_coupling(self, channel1, channel2, t_min, t_max, start_idx=None, end_idx=None):
        """Calculate phase-amplitude coupling (PAC) between two channels within a specified time interval.

        Args:
            channel1 (int): Index of the first channel.
            channel2 (int): Index of the second channel.
            t_min (float): Start time of the interval of interest in seconds.
            t_max (float): End time of the interval of interest in seconds.
            start_idx (int, optional): Start index of the interval in the data. Defaults to None.
            end_idx (int, optional): End index of the interval in the data. Defaults to None.

        Returns:
            modulation_index (float): The phase-amplitude coupling modulation index.

        """
        if start_idx is None:
            start_idx = np.argmin(np.abs(self.time - t_min))
        if end_idx is None:
            end_idx = np.argmin(np.abs(self.time - t_max))

        x = self.data[:, channel1, start_idx:end_idx]
        y = self.data[:, channel2, start_idx:end_idx]

        # Calculate the phase of the low-frequency signal
        analytic_signal = hilbert(x)
        phase = np.angle(analytic_signal)

        # Calculate the amplitude envelope of the high-frequency signal
        analytic_signal = hilbert(y)
        amplitude_envelope = np.abs(analytic_signal)

        # Calculate the mean amplitude for each phase bin
        phase_bins = np.linspace(-np.pi, np.pi, 21)
        bin_indices = np.digitize(phase, phase_bins) - 1
        mean_amplitude = [np.mean(amplitude_envelope[bin_indices == k]) for k in range(len(phase_bins))]

        # Calculate the PAC using the modulation index
        mean_amplitude = np.array(mean_amplitude)
        modulation_index = (np.max(mean_amplitude) - np.min(mean_amplitude)) / (np.max(mean_amplitude) + np.min(mean_amplitude))

        return modulation_index

    def phase_amplitude_coupling_matrix(self, t_min, t_max):
        """
        Calculates the phase-amplitude coupling (PAC) between all pairs of channels within a specific time interval.

        Args:
            t_min (float): The start time of the interval of interest in seconds.
            t_max (float): The end time of the interval of interest in seconds.

        Returns:
            pac (ndarray): The phase-amplitude coupling matrix of shape (num_channel, num_channel) for the given time interval.

        """
        # Get the indices of the samples in the time interval
        start_idx = np.argmin(np.abs(self.time - t_min))
        end_idx = np.argmin(np.abs(self.time - t_max))

        # Calculate the PAC for all pairs of channels
        pac = np.zeros((self.num_channel, self.num_channel))
        for i in range(self.num_channel):
            for j in range(i + 1, self.num_channel):
                modulation_index = self.phase_amplitude_coupling(i, j, t_min, t_max, start_idx=start_idx, end_idx=end_idx)
                pac[i, j] = modulation_index
                pac[j, i] = modulation_index

        return pac

