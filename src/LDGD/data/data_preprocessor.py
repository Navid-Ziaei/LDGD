import numpy as np
from scipy import signal
from tqdm import tqdm
import mne


class PreProcessing:
    def __init__(self, dataset):
        self.time = dataset.time
        self.labels = dataset.label
        self.dataset = dataset
        self.data = dataset.data
        self.fs = dataset.fs
        self.channel_names = dataset.channel_name
        self.n_trials, self.n_channels, self.n_samples = self.data.shape

        # TODO: Add time, labels and data as property and add setter for them

    def remove_base_line(self, baseline_t_min, baseline_t_max):
        # Determine start and end indices for the baseline time window
        idx_start = np.argmin(np.abs(self.time - baseline_t_min))
        idx_end = np.argmin(np.abs(self.time - baseline_t_max))

        if (idx_end - idx_start) > 5:
            # Calculate the baseline mean values for each trial and channel
            baseline_means = np.mean(self.data[:, :, idx_start:idx_end], axis=2, keepdims=True)

            # Subtract the baseline mean from all time points for each trial and channel
            eeg_data_baseline_removed = self.data - baseline_means
            self.dataset.data = eeg_data_baseline_removed
            self.data = eeg_data_baseline_removed

        return self.data

    def resample(self, f_resample, anti_alias_filter=False):
        # chane data into mne epochs
        epochs = self.array_to_mne_epoch()

        decim_factor = int(self.fs / f_resample)  # Replace this with your desired decimation factor

        if anti_alias_filter is True:
            # re-sample first filter data to prevent aliasing
            epochs_down_sampled = epochs.copy().resample(f_resample)
        else:
            epochs_down_sampled = epochs.copy().decimate(decim_factor)

        self.data = epochs_down_sampled.get_data()

        self.time = self.time[0] + epochs_down_sampled.times
        self.fs = epochs_down_sampled.info['sfreq']
        self.dataset.fs = epochs_down_sampled.info['sfreq']
        self.dataset.data = self.data
        self.dataset.time = self.time

        return self.time, self.data

    def common_average_referencing(self):
        # Compute the mean signal across all electrodes for each time point
        mean_signal = np.mean(self.data, axis=1, keepdims=True)
        # Subtract the mean signal from each electrode's signal
        self.data = self.data - mean_signal
        self.dataset.data = self.data

        return self.data

    def filter_data(self, low_cutoff=0.1, high_cutoff=300):
        # Compute filter coefficients
        nyquist_freq = self.fs / 2
        b, a = signal.butter(4, [low_cutoff / nyquist_freq, high_cutoff / nyquist_freq], btype='bandpass')

        # Filter iEEG data
        filtered_data = signal.filtfilt(b, a, self.data, axis=2)

    def filter_data_mne(self, low_cutoff=0.1, high_cutoff=300, filter_length='auto',
                        l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                        method='fir', iir_params=None, phase='zero',
                        fir_window='hamming', fir_design='firwin', use_raw=False):

        # Create MNE RawArray object
        if use_raw is True:
            data_mne = self.array_to_mne_raw()
        else:
            data_mne = self.array_to_mne_epoch()

        # Filter iEEG data
        raw_filtered = data_mne.filter(l_freq=low_cutoff, h_freq=high_cutoff, filter_length=filter_length,
                                       l_trans_bandwidth=l_trans_bandwidth, h_trans_bandwidth=h_trans_bandwidth,
                                       method=method, iir_params=iir_params, phase=phase,
                                       fir_window=fir_window, fir_design=fir_design)

        # Get filtered iEEG data as a NumPy array
        filtered_data = raw_filtered.get_data()

        if use_raw is True:
            # Reshape filtered data back to its original shape
            self.data = filtered_data.reshape(self.n_channels, self.n_trials, self.n_samples).transpose(1, 0, 2)
        else:
            self.data = filtered_data

        self.dataset.data = self.data
        return self.data

    def remove_line_noise(self,
                          freqs_to_remove=[60],
                          filter_length='auto',
                          phase='zero',
                          method='fir',
                          trans_bandwidth=1.0,
                          mt_bandwidth=None):
        # Reshape data to 2D array of shape (n_channels, n_samples)
        data_mne = self.array_to_mne_raw()

        # Apply the notch filter
        notch_filtered_epochs = data_mne.notch_filter(freqs_to_remove,
                                                      filter_length=filter_length,
                                                      phase=phase,
                                                      method=method,
                                                      trans_bandwidth=trans_bandwidth,
                                                      mt_bandwidth=mt_bandwidth)

        filtered_data = notch_filtered_epochs.get_data()

        self.data = filtered_data.reshape(self.n_channels, self.n_trials, self.n_samples).transpose(1, 0, 2)
        self.dataset.data = self.data
        return self.data

    def array_to_mne_raw(self):
        ieeg_data = self.data.transpose(1, 0, 2).reshape(self.n_channels, -1)

        # Create MNE RawArray object
        info = mne.create_info(ch_names=self.channel_names, sfreq=self.fs, ch_types='seeg')
        raw = mne.io.RawArray(ieeg_data, info)

        return raw

    def mne_raw_to_array(self, raw):
        # Get filtered iEEG data as a NumPy array
        data = raw.get_data()

        # Reshape filtered data back to its original shape
        data = data.reshape(self.n_channels, self.n_trials, self.n_samples).transpose(1, 0, 2)
        return data

    def array_to_mne_epoch(self, event_id=None):
        # Create an MNE Info object with the channel information
        info = mne.create_info(ch_names=self.channel_names,
                               sfreq=self.fs,
                               ch_types=['seeg'] * self.n_channels)

        # Event Id
        if event_id is None:
            event_id = np.arange(self.n_trials)
        # Create the events array with shape (n_trials, 3)
        events = np.column_stack((event_id, np.zeros(self.n_trials, dtype=int), np.zeros(self.n_trials, dtype=int)))

        # Create an EpochsArray object with the given data, time vector, and labels
        epochs = mne.EpochsArray(self.data, info, events=events, event_id=None)

        return epochs

    def baseline_removal(self, method='zero_mean', window_size=100, min_time=0, max_time=None):
        """
        Remove baseline drifts from data using one of several methods.

        Parameters
        ----------
        method : str, optional
            The baseline removal method to use. Defaults to 'zero_mean'.
            Options are: 'zero_mean', 'mean_norm', 'poly_fit', 'mov_avg', 'median'.
        window_size : int, optional
            The size of the window to use for the moving average or median filter methods.
            Defaults to 100.
        min_time : int, optional
            The minimum time index of the pre-stimulus period. Defaults to 0.
        max_time : int or None, optional
            The maximum time index of the pre-stimulus period. Defaults to None.

        Returns
        -------
        data_out : numpy array
            The baseline-removed data, of shape (n_epochs, n_channels, n_samples).
        """
        data = self.data

        if min_time is not None:
            min_time_idx = np.min(np.where(self.time > min_time))
        else:
            min_time_idx = 0

        if max_time is not None:
            max_time_idx = np.max(np.where(self.time < max_time))
        else:
            max_time_idx = len(self.time)

        # Calculate mean and standard deviation of the data for the pre-stimulus period
        data_mean = np.mean(data[:, :, min_time_idx:max_time_idx], axis=-1, keepdims=True)
        data_std = np.std(data[:, :, min_time_idx:max_time_idx], axis=-1, keepdims=True)

        # Apply baseline removal method
        if method == 'zero_mean':
            data_out = data - data_mean
        elif method == 'mean_norm':
            data_out = data / data_mean
        elif method == 'poly_fit':
            data_out = signal.detrend(data, axis=-1, type='linear')
        elif method == 'mov_avg':
            window = signal.windows.hann(window_size)
            data_out = signal.convolve(data, window[None, None, :], mode='same', axis=-1) / np.sum(window)
            data_out -= data_mean
        elif method == 'median':
            data_out = signal.medfilt(data, kernel_size=(1, 1, window_size))
            data_out -= data_mean
        else:
            raise ValueError("Invalid baseline removal method.")

        # Normalize data by standard deviation
        data_out /= data_std
        self.data = data_out
        return self.data


def moving_window_fft(data, data_info, window_size_ms, t_min, t_max, overlap_percent=50):
    """
        Calculates the power spectral density (PSD) of the signal within a sliding window using Welch's method and
        returns the band power within specific frequency bands.

        :param data: np.array of shape (n_trial, n_channel, n_samples)
        :param data_info: sampling rate of the data
        :param window_size_ms: size of sliding window in milliseconds
        :param overlap_percent: percentage of overlap between consecutive windows (0 to 100)
        :param t_min: start time of the sliding window in seconds
        :param t_max: end time of the sliding window in seconds
        :return: band_power: dictionary containing band power for each channel and trial
    """
    fs = data_info['fs']
    time = data_info['time']
    # Convert window size and overlap to sample indices
    window_size = int(window_size_ms * fs / 1000)
    overlap = int(window_size * overlap_percent / 100)
    step = window_size - overlap
    # Convert time interval to sample indices
    t_start = np.argmin(np.abs(time - t_min))
    t_end = np.argmin(np.abs(time - t_max))
    num_samples = t_end - t_start
    num_sample_after_ovfft = int(1 + (num_samples - window_size) / (window_size - overlap))
    # Initialize frequency bands of interest
    freq_bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
        'gamma': (30, 100),
        'high_gamma': (100, 200)
    }

    # Initialize output variables
    band_power = {band: np.zeros((data.shape[0], data.shape[1], num_sample_after_ovfft)) for band in freq_bands}
    time_vect = np.zeros(num_sample_after_ovfft)
    # Loop over trials and channels
    for trial in tqdm(range(data.shape[0])):
        for channel in range(data.shape[1]):
            # Initialize window parameters
            window_start = t_start
            window_end = t_start + window_size
            idx = 0
            # Loop over windows
            while window_end <= t_end:
                # Extract data in the current window
                data_window = data[trial, channel, window_start:window_end]

                # Calculate the power spectral density using Welch's method
                f, psd = signal.welch(data_window, fs)

                # Loop over frequency bands of interest and calculate band power
                for band, freq_range in freq_bands.items():
                    band_start = np.argmin(np.abs(f - freq_range[0]))
                    band_end = np.argmin(np.abs(f - freq_range[1]))
                    band_power[band][trial, channel, idx] = np.sum(psd[band_start:band_end])

                # Append the current time to the time vector
                time_vect[idx] = time[window_end]

                # Increment the window parameters
                window_start += step
                window_end += step
                idx += 1

    return band_power, time_vect
