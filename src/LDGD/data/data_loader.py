import warnings
import os
import tarfile
import urllib
# import mne
# import mne_bids

from collections import OrderedDict
from src.gp_project_pytorch.utils import resample, smooth_signal, zscore
from abc import ABC, abstractmethod
from scipy import io
from tqdm import tqdm
from pathlib import Path

from scipy.signal import butter, filtfilt, hilbert, decimate, firwin, freqz
from sklearn.datasets import load_iris, load_wine
from sklearn.datasets import fetch_openml

from sklearn.model_selection import train_test_split

from src import *
from sklearn.datasets import make_moons, make_circles, make_classification
import matplotlib.pyplot as plt


def generate_data(pattern, n_samples=100, noise=0.1, n_features=50, increase_method='add_noise'):
    if pattern == 'moon':
        X, y = make_moons(n_samples=n_samples, noise=noise)
    elif pattern == 'ring':
        X, y = make_circles(n_samples=n_samples, noise=noise)
    elif pattern == 'cross':
        X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0,
                                   n_clusters_per_class=1)
    else:
        raise ValueError("Invalid pattern")

    increase_method = 'linear'
    # add noise dimensions
    if increase_method == 'add_noise':
        noise_dims = np.random.rand(n_samples, n_features - 2)  # adjust to match desired total dimensionality
        X = np.concatenate((X, noise_dims), axis=1)
    elif increase_method == 'linear':
        transform_matrix = np.random.rand(X.shape[-1], n_features)  # adjust to match desired total dimensionality

        color_list = ['r', 'b', 'g']
        label_list = ['clasa 1', 'class 2']
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.scatter(X[y == 0, 0], X[y == 0, 1],
                   c='r', s=40, alpha=1,
                   edgecolor='r')  # edgecolor adds a border to the markers
        ax.scatter(X[y == 1, 0], X[y == 1, 1],
                   c='b', s=40, alpha=1,
                   edgecolor='b')
        # Adding labels with larger font size
        ax.set_xlabel('$X_1$', fontsize=32)
        ax.set_ylabel('$X_2$', fontsize=32)

        # Setting tick label sizes
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)

        # Adjusting spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)

        # Optional: Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

        # Optional: Set aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Optional: Add legend
        # If you have a legend to add, uncomment and modify the following line
        plt.legend(['Class 1', 'Class 2'], fontsize=24)  # Replace labels with appropriate ones
        # Improve layout
        plt.tight_layout()
        fig.savefig("dataset.png")
        fig.savefig("dataset.svg")

        # Show the plot
        plt.show()
        X = np.matmul(X, transform_matrix)
        noise_dims = np.random.rand(n_samples, X.shape[-1])  # adjust to match desired total dimensionality
        X = np.concatenate((X, noise_dims), axis=1)
    else:
        raise ValueError("The increase dimension method is not defind")

    return X, y


def load_dataset(dataset_name, test_size=0.1, **kwargs):
    supported_datasets = ['oil', 'wine', 'iris', 'usps', 'synthetic']
    if dataset_name.lower() == 'oil':
        wdir = Path(os.path.abspath('')).parent.parent
        os.chdir(wdir)

        url = "http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/resources/3PhData.tar.gz"
        urllib.request.urlretrieve(url, '3PhData.tar.gz')
        with tarfile.open('3PhData.tar.gz', 'r') as f:
            f.extract('DataTrn.txt')
            f.extract('DataTrnLbls.txt')
        X = np.loadtxt(fname='DataTrn.txt')
        label = np.loadtxt(fname='DataTrnLbls.txt')
        y = (torch.tensor(label) @ np.diag([0, 1, 2])).sum(axis=1)
        y = y.detach().numpy()

    elif dataset_name.lower() == 'iris':
        iris = load_iris()
        X, y = iris.data, iris.target

    elif dataset_name.lower() == 'wine':
        wine = load_wine()
        X, y = wine.data, wine.target

    elif dataset_name.lower() == 'usps':
        usps = fetch_openml('usps', version=1)
        usps_data = usps['data']
        usps_target = usps['target']
        X = usps_data.values
        y = usps_target.values - 1
    elif dataset_name.lower() == 'synthetic':
        # Extract parameters for synthetic data generation
        pattern = kwargs.get('pattern', 'moon')  # default pattern
        n_samples = kwargs.get('n_samples', 500)
        noise = kwargs.get('noise', 0.1)
        n_features = kwargs.get('n_features', 10)
        increase_method = kwargs.get('increase_method', 'linear')

        # Generate synthetic data
        X, y = generate_data(pattern, n_samples, noise, n_features, increase_method)
    else:
        raise ValueError(f"Dataset should be from {supported_datasets}")

    # One-hot encode the labels
    y_one_hot = np.zeros((y.shape[0], len(np.unique(y))))
    y_one_hot[np.arange(y.shape[0]), np.uint(y)] = 1

    # Split the dataset into training and test sets
    X_train, X_test, y_train, y_test, y_train_labels, y_test_labels = train_test_split(X, y_one_hot, y,
                                                                                       test_size=test_size,
                                                                                       random_state=42)
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    y_train_labels_tensor = torch.tensor(y_train_labels)
    y_test_labels_tensor = torch.tensor(y_test_labels)

    return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor, y_train_labels_tensor, y_test_labels_tensor


class AbstractDataLoader(ABC):
    def __init__(self, paths, settings):
        self.target_class = settings.target_class
        self.patient = settings.patient
        self.experiment = settings.experiment
        self.task = settings.task  # 'FlickerShapes' or 'Flicker'
        self.paths = paths
        self.file_format = settings.file_format
        self.results = {}

    @abstractmethod
    def read_data(self, patient_dir_path, file_dir):
        pass

    def load_data(self):
        data_list = []
        if isinstance(self.task, str):
            task = [self.task]
        if isinstance(self.patient, str):
            patient = [self.patient]

        file_dir = [[], [], []]
        for task_dir in os.listdir(self.paths.prepared_dataset_path):
            if task and task_dir.replace('_task', '') not in task:
                continue
            file_dir[0] = task_dir
            task_dir_path = os.path.join(self.paths.prepared_dataset_path, task_dir)
            if os.path.isdir(task_dir_path):
                for patient_dir in os.listdir(task_dir_path):
                    if patient and patient_dir not in patient:
                        continue
                    file_dir[1] = patient_dir
                    patient_dir_path = os.path.join(task_dir_path, patient_dir)
                    data_list.extend(self.read_data(patient_dir_path=patient_dir_path,
                                                    file_dir=file_dir))
        if len(data_list) == 0:
            raise ValueError(f" No data found for patient {self.patient} and task {self.task} in "
                             f"{self.paths.prepared_dataset_path}")
        return data_list

    def get_available_blocks(self):
        pass


class iEEGDataLoader(AbstractDataLoader):
    def __init__(self, paths, settings):
        super().__init__(paths, settings)
        # Add any additional initialization if needed

    def read_data(self, patient_dir_path, file_dir):
        i = 0
        data_ieeg_list = []
        for file in os.listdir(patient_dir_path):
            if file.endswith('.pkl'):
                file_dir[2] = file
                file_path = os.path.join(patient_dir_path, file)
                data = pd.read_pickle(file_path)
                print(f"{i}: Data {file} \t Block Number: {data['trial_info']['BlockNumber'].unique()} \t"
                      f"Number of trials: {data['trial_info'].shape[0]}")
                nan_trials = list(np.unique(np.where(np.isnan(data['data']))[0]))
                if len(nan_trials) > 0:
                    print(f"Warning: Trial {nan_trials} contains NaN. It is removed")
                    for trial_idx in nan_trials:
                        data['data'] = np.delete(data['data'], trial_idx, axis=0)
                        data['trial_info'] = data['trial_info'].drop(trial_idx, axis=0)
                        data['index'] = np.delete(data['index'], trial_idx, axis=0)
                        data['label'] = data['label'].drop(trial_idx, axis=0)

                data_ieeg = IEEGData()
                data_ieeg.dict_to_iEEG_format(data_dict=data, meta=file_dir)

                i = i + 1
                data_ieeg_list.append(data_ieeg)
        return data_ieeg_list


class AUDIOVISDataLoader:
    def __init__(self, paths, settings, **kwargs):
        self.paths = paths
        self.settings = settings
        self.subject = settings.patient
        self.task = 'film'
        self.datatype = 'ieeg'
        self.acquisition = 'clinical'
        self.root = paths.prepared_dataset_path
        self.expected_duration = 390
        self.__dict__.update(kwargs)

    def load_data(self):
        self._set_paths()
        self._load_data()
        self._get_bad_electrodes()
        self.preprocess()
        self.extract_events(plot=False)
        epochs, data, target_category = self._get_epochs(tmin=-1, tmax=20)
        trial_info = self.raw_car.annotations.to_data_frame()
        trial_info = trial_info[2:-1]
        trial_info['Trial number'] = np.arange(trial_info.shape[0])
        data_ieeg = IEEGData()
        data_ieeg.data = data
        data_ieeg.channel_name = self.raw_car.info['ch_names']
        data_ieeg.trial_info = trial_info
        data_ieeg.data_info = self.raw_car.info
        data_ieeg.time = epochs.times  # time should be in second
        data_ieeg.label = target_category - 1
        data_ieeg.fs = self.raw_car.info['sfreq']
        data_ieeg.meta = ['sub-' + self.subject, self.task, self.session, self.raw_car.filenames]

        return [data_ieeg]

    def _get_epochs(self, tmin=-0.1, tmax=20):
        baseline = (None, 0)  # means from the start of the epoch to event onset. Adjust if needed.

        epochs = mne.Epochs(self.raw_car, self.events, event_id=self.event_id, tmin=tmin, tmax=tmax,
                            baseline=baseline, preload=True)
        x = epochs.get_data()  # shape is (n_trials, n_channels, n_times)
        y = epochs.events[:, 2]  # The event ID associated with each epoch/trial

        return epochs, x, y

    def _set_paths(self):
        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                      task=self.task,
                                      suffix='ieeg',
                                      extension='.vhdr',
                                      datatype=self.datatype,
                                      acquisition=self.acquisition,
                                      root=self.root)

        print(self.task)
        assert len(bids_path.match()) == 1, 'None or more than one run for task is found'

        self.raw_path = str(bids_path.match()[0])
        self.run = mne_bids.get_entities_from_fname(bids_path.match()[0])['run']
        self.session = mne_bids.get_entities_from_fname(bids_path.match()[0])['session']

        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                      task=self.task,
                                      session=self.session,
                                      suffix='channels',
                                      run=self.run,
                                      extension='.tsv',
                                      datatype=self.datatype,
                                      acquisition=self.acquisition,
                                      root=self.root)
        self.channels_path = str(bids_path)

        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                      session=self.session, suffix='electrodes',
                                      extension='.tsv',
                                      datatype=self.datatype,
                                      acquisition=self.acquisition,
                                      root=self.root)
        self.electrodes_path = str(bids_path)

        bids_path = mne_bids.BIDSPath(subject=self.subject,
                                      suffix='T1w',
                                      extension='.nii.gz',
                                      root=self.root)
        self.anat_path = str(bids_path.match()[0])
        self.anat_session = mne_bids.get_entities_from_fname(bids_path.match()[0])['session']
        print('Picking up BIDS files done')

    def _load_data(self):
        self.raw = mne.io.read_raw_brainvision(self.raw_path,
                                               eog=(['EOG']),
                                               misc=(['OTHER', 'ECG', 'EMG']),
                                               scale=1.0,
                                               preload=False,
                                               verbose=True)
        self.channels = pd.read_csv(self.channels_path, sep='\t', header=0, index_col=None)
        self.electrodes = pd.read_csv(self.electrodes_path, sep='\t', header=0, index_col=None)
        self.raw.set_channel_types({ch_name: str(x).lower()
        if str(x).lower() in ['ecog', 'seeg', 'eeg'] else 'misc'
                                    for ch_name, x in zip(self.raw.ch_names, self.channels['type'].values)})
        self.other_channels = self.channels['name'][~self.channels['type'].isin(['ECOG', 'SEEG'])].tolist()
        self.raw.drop_channels(self.other_channels)
        print('Loading BIDS data done')

    def _get_bad_electrodes(self):
        self.bad_electrodes = self.channels['name'][
            (self.channels['type'].isin(['ECOG', 'SEEG'])) & (self.channels['status'] == 'bad')].tolist()
        self.all_electrodes = self.channels['name'][(self.channels['type'].isin(['ECOG', 'SEEG']))].tolist()
        print('Getting bad electrodes done')

    def preprocess(self):
        raw_dbe = self._discard_bad_electrodes()
        raw_nf = self._notch_filter()
        raw_car = self._common_average_reference()
        return raw_car

    def _discard_bad_electrodes(self):
        if self.raw is not None:
            [self.bad_electrodes.remove(i) for i in self.bad_electrodes if i not in self.raw.ch_names]
            self.raw.info['bads'].extend([ch for ch in self.bad_electrodes])
            print('Bad channels indicated: ' + str(self.bad_electrodes))
            print('Dropped ' + str(self.raw.info['bads']) + 'channels')
            self.raw.drop_channels(self.raw.info['bads'])
            self.raw.load_data()
            print('Remaining channels ' + str(self.raw.ch_names))
            return self.raw

    def _notch_filter(self):
        if self.raw is not None:
            if np.any(np.isnan(self.raw._data)):
                # self.raw._data[np.isnan(self.raw._data)]=0 # bad hack for nan values
                warnings.warn('There are NaNs in the data, replacing with 0')
            self.raw.notch_filter(freqs=np.arange(50, 251, 50), verbose=False)
            print('Notch filter done')
        return self.raw

    def _common_average_reference(self):
        if self.raw is not None:
            self.raw_car, _ = mne.set_eeg_reference(self.raw.copy(), 'average')
            print('CAR done')
        return self.raw_car

    def extract_events(self, plot=False):
        events, events_id = self._read_events()
        if plot:
            self._plot_events()
        return events, events_id

    def _read_events(self):
        if self.raw is not None:
            if self.task == 'film':
                custom_mapping = {'Stimulus/music': 2,
                                  'Stimulus/speech': 1,
                                  'Stimulus/end task': 5}
            elif self.task == 'rest':
                custom_mapping = {'Stimulus/start task': 1,
                                  'Stimulus/end task': 2}
            else:
                raise NotImplementedError
            self.events, self.event_id = mne.events_from_annotations(self.raw, event_id=custom_mapping,
                                                                     use_rounding=False)
            print('Reading events done')
        return self.events, self.event_id

    def _plot_events(self):
        if self.raw_car is not None:
            self.raw_car.plot(events=self.events,
                              start=0,
                              duration=180,
                              color='gray',
                              event_color={2: 'g', 1: 'r', 5: 'b'},
                              bgcolor='w')

    def extract_bands(self, apply_hilbert, smooth=False):
        if self.raw_car is not None:
            self._compute_band_envelopes(apply_hilbert)
            self._crop_band_envelopes()
            self._resample_band_envelopes()
            if smooth:
                self._smooth_band_envelopes()
            bands5, bands6, band_block_means = self._compute_block_means_per_band()
        return bands5

    def _compute_band_envelopes(self, apply_hilbert):
        if self.raw_car is not None:
            bands = {'delta': [1, 4], 'theta': [5, 8], 'alpha': [8, 12], 'beta': [13, 24], 'gamma': [60, 120]}
            self.bands1 = OrderedDict.fromkeys(bands.keys())
            for key in self.bands1.keys():
                if apply_hilbert:
                    self.bands1[key] = self.raw_car.copy().filter(bands[key][0], bands[key][1],
                                                                  verbose=False).apply_hilbert(
                        envelope=True).get_data().T
                else:
                    self.bands1[key] = self.raw_car.copy().filter(bands[key][0], bands[key][1],
                                                                  verbose=False).get_data().T
            print('Extracting band envelopes done')
        return self.bands1

    def _crop_band_envelopes(self):
        if self.bands1 is not None:
            self.bands2 = OrderedDict.fromkeys(self.bands1.keys())
            for key in self.bands1.keys():
                self.bands2[key] = self.bands1[key][self.events[0, 0]:self.events[-1, 0]]
        return self.bands2

    def _resample_band_envelopes(self):
        if self.bands1 is not None:
            self.bands3 = OrderedDict.fromkeys(self.bands1.keys())
            for key in self.bands1.keys():
                self.bands3[key] = resample(self.bands2[key], 25, int(self.raw.info['sfreq']))
        return self.bands3

    def _smooth_band_envelopes(self):
        if self.bands1 is not None:
            for key in self.bands1.keys():
                self.bands1[key] = np.apply_along_axis(smooth_signal, 0, self.bands1[key], 5)

    def _compute_block_means_per_band(self):
        if self.bands1 is not None:
            self.band_block_means = OrderedDict.fromkeys(self.bands1.keys())
            band6 = OrderedDict.fromkeys(self.bands1.keys())
            for key in self.bands1.keys():
                band6[key] = zscore(self.bands3[key][:self.expected_duration * 25])
                band7 = band6[key].reshape((-1, 750, band6[key].shape[-1]))  # 13 blocks in chill or 6 in rest
                self.band_block_means[key] = np.mean(band7, 1)
        return band6, band7, self.band_block_means


class OCEDEEGDataLoader(AbstractDataLoader):
    def __init__(self, paths, settings):
        super().__init__(paths, settings)
        # Add any additional initialization if needed

    def read_data(self, patient_dir_path, file_dir):
        i = 0
        for file in os.listdir(patient_dir_path):
            if file.endswith(self.file_format):
                file_dir[2] = file
                file_path = os.path.join(patient_dir_path, file)
                patient_data_mat = io.loadmat(file_path)
                # extract data from mat file
                fs = patient_data_mat['fs'][0][0]
                target_example = np.squeeze(patient_data_mat['exemplarLabels']) - 1
                target_category = np.squeeze(patient_data_mat['categoryLabels']) - 1
                data = patient_data_mat['xEpoched'][:-1]
                num_electrodes, num_samples, num_trials = data.shape
                data = np.transpose(data, (2, 0, 1))
                time = np.squeeze(patient_data_mat['epochTimeSamples'])
                unique_categories = np.unique(target_category) - 1
                onsets = np.squeeze(patient_data_mat['onsets'])
                trial_info = {'Trial number': list(np.arange(0, num_trials)),
                              'Block number': patient_data_mat['sessionID'][0],
                              'exemplar labels': target_example,
                              'category labels': target_category,
                              'onsets': onsets}

                print(f"{i}: Data {file} \t session ID: {patient_data_mat['sessionID'][0]} \t"
                      f"Number of trials: {data.shape[0]}")

                channel_name = list(np.arange(0, num_electrodes))
                data_ieeg = IEEGData()
                data_ieeg.data = data
                data_ieeg.channel_name = [str(ch) for ch in channel_name]
                data_ieeg.trial_info = pd.DataFrame(trial_info)
                data_ieeg.data_info = trial_info
                data_ieeg.time = time / 1000  # time should be in second
                data_ieeg.label = target_category
                data_ieeg.fs = fs
                data_ieeg.meta = file_dir

                i = i + 1
        return data_ieeg


class CustomDataLoader(AbstractDataLoader, ABC):
    def __init__(self, paths, settings):
        super().__init__(paths, settings)
        # Add any additional initialization if needed

    def get_available_blocks(self):
        # Implement the logic here
        pass

    # If you need to override or extend the load_data method, you can do so:
    def load_data(self):
        data_list = super().load_data()
        # Add any additional logic if needed
        return data_list

    # You can add additional methods specific to this class as well.


class IEEGData:
    def __init__(self, data=None, fs=None):
        self.trial_time_annotation = None
        self.time_annotation = None
        self.continuous_signal = None
        self.continuous_indicator = None
        self.data = data
        self.channel_name = None
        self.trial_info = None
        self.data_info = None
        self.time = None
        self.label = None
        self.fs = fs
        self.meta = None

    def dict_to_iEEG_format(self, data_dict, meta):
        self.data = data_dict['data']
        self.fs = data_dict['fs']
        self.time = data_dict['time']
        self.channel_name = data_dict['channel_names']
        self.trial_info = data_dict['trial_info']
        self.label = data_dict['label']
        self.meta = meta

    def epoched_to_continuous(self, settings, debug=False, save_path=None):
        y = self.label.values
        time = self.time * 1000
        n_trials, n_channels, _ = self.data.shape

        self.trial_info = self.trial_info.loc[:, ~self.trial_info.columns.duplicated()]

        if self.trial_time_annotation is None:
            self.get_time_annotation(settings=settings)
        continuous_signal = self.data[0]
        continuous_indicator = self._create_trial_indicator(task=settings.task,
                                                            label=y,
                                                            trial_idx=0)
        continuous_indicator_list = [continuous_indicator]
        time_vecror = []
        for i in tqdm(range(n_trials - 1)):
            trial = self.data[i + 1]
            trial_indicator = self._create_trial_indicator(task=settings.task,
                                                           label=y,
                                                           trial_idx=i + 1)
            for t in range(1, trial.shape[-1]):
                if np.all(continuous_signal[[0, 1, 10], -t:] == trial[[0, 1, 10], :t]):
                    # print(10000 - t)
                    time_vecror.append(10000 - t)
                    if debug is True:
                        if np.sum(continuous_signal[:, -t] - trial[:, 0]) != 0:
                            print(
                                f"Misalignment from trial {i} to {i + 1} in channels {np.where(continuous_signal[:, -t] != trial[:, 0])[0]}")
                    continuous_signal = np.concatenate((continuous_signal[:, :-t], trial), axis=1)
                    continuous_indicator = np.concatenate((
                        np.zeros_like(continuous_indicator[:-t]) * np.nan,
                        trial_indicator), axis=0)
                    continuous_indicator_list.append(continuous_indicator)
                    break
            if t == trial.shape[-1] - 1:
                print(f"error: trial {i} to {i + 1} alignment not found")
            all_indicator = [
                list(continuous_indicator) + [np.nan] * (continuous_signal.shape[-1] - len(continuous_indicator)) for
                continuous_indicator in continuous_indicator_list]

        all_indicator = np.stack(all_indicator)
        fig, ax = plt.subplots(1, figsize=(12, 10))
        sns.heatmap(all_indicator, ax=ax)
        if save_path is None:
            plt.show()
        else:
            fig.savefig(save_path + f"continuous_signals_annotation.png")
        final_indicator = np.ones_like(all_indicator[0]) * np.nan
        for t in tqdm(range(all_indicator.shape[-1])):
            if any(all_indicator[:, t] == 0):
                final_indicator[t] = 0
            elif len(np.unique(all_indicator[:, t])) == 2:
                if np.unique(all_indicator[:, 10000])[0] is not np.nan:
                    final_indicator[t] = np.unique(all_indicator[:, t])[0]
                else:
                    final_indicator[t] = np.unique(all_indicator[:, t])[1]
            elif len(np.unique(all_indicator[:, t])) > 2:
                trial_idxs = []
                for uniqevalue in list(np.unique(all_indicator[:, t])):
                    if uniqevalue is not np.nan:
                        trial_idxs.extend(list(np.where(all_indicator[:, t] == uniqevalue)[0]))
                final_indicator[t] = all_indicator[np.max(trial_idxs), t]

        self.continuous_indicator = final_indicator
        self.continuous_signal = continuous_signal

        if debug is True:
            diff = np.diff(final_indicator)
            change_indices = np.where(diff != 0)[0]
            change_indices += 1
            start_points = [idx for idx in change_indices if final_indicator[idx - 1] == 0]
            print(np.diff(start_points))

        return continuous_signal, continuous_indicator

    def get_time_annotation(self, settings):
        if settings.task == 'imagine':
            # Get columns with duplicated names
            self.trial_info = self.trial_info.iloc[:, ~self.trial_info.columns.duplicated(keep='first')]

            imagination_period_end = (
                    self.trial_info['Onset Choice Period'] - self.trial_info['Fixation Onset NEV']).values
            trial_end = (self.trial_info['Trial offset'] - self.trial_info['Fixation Onset NEV']).values
            choice_period_end = (self.trial_info['Image Onset NEV'] - self.trial_info['Fixation Onset NEV']).values
            reaction_time = (self.trial_info['Response time NEV'] - self.trial_info['Fixation Onset NEV']).values
            if any(imagination_period_end < 1) or any(imagination_period_end > 7):
                imagination_period_end = (self.trial_info['CrossStart'] - self.trial_info['StartTrial']).values
            if any(np.isnan(trial_end)) or any(trial_end < 0):
                trial_end = (self.trial_info['TimeTrial'] - self.trial_info['StartTrial']).values
            if any(np.isnan(choice_period_end)):
                choice_period_end[np.isnan(choice_period_end)] = 5.5
                print("Warning: NaN in Image Onset NEV")

            self.time_annotation = {
                "imagination onset": 0,
                "choice period onset": np.round(np.mean(imagination_period_end) * 1000),
                "image onset": np.round(np.mean(choice_period_end) * 1000),
                "trial end": np.round(np.mean(trial_end) * 1000)
            }

            self.trial_time_annotation = {
                'imagination_period_end': imagination_period_end * 1000,
                'reaction_time': reaction_time * 1000,
                'choice_period_end': choice_period_end * 1000,
                'trial end': trial_end * 1000
            }
        elif settings.task == 'flicker':
            fixation_onset = self.trial_info['Fixation Onset NEV'] - self.trial_info['Image Onset NEV']
            if 'Image Offset NEV' not in self.trial_info.columns:
                self.trial_info['Image Offset NEV'] = 0.0
                self.trial_info['Image Offset NEV'].values[:-1] = self.trial_info['Fixation Onset NEV'].values[1:]
            trial_end = self.trial_info['Image Offset NEV'] - self.trial_info['Image Onset NEV']

            if any(trial_end < 0.1) or any(trial_end > 5):
                print(
                    f"trials with trial_end annotation problem {np.where((trial_end < 0.1) | (trial_end > 5))[0]}")
                trial_end[(trial_end < 0.1) | (trial_end > 5)] = 1

            image_onset = np.zeros(self.trial_info.shape[0])
            if settings.patient in ['p01', 'p03', 'p04']:
                trial_end = np.zeros(self.trial_info.shape[0])
                fixation_onset = -self.trial_info['Image Onset NEV'] + self.trial_info['Fixation Onset NEV']
                image_onset = fixation_onset

            self.time_annotation = {
                "fixation onset": np.round(np.mean(fixation_onset) * 1000),
                "image onset": np.round(np.mean(image_onset) * 1000),
                "trial end": np.round(np.mean(trial_end) * 1000)
            }

            self.trial_time_annotation = {
                "fixation onset": np.round(fixation_onset * 1000),
                "image onset": np.round(image_onset * 1000),
                "trial end": np.round(trial_end * 1000)
            }
        elif settings.task == 'flicker_shape':
            if settings.patient in ['p10', 'p11', 'p14']:
                trial_end = self.trial_info['Image Offset NEV'] - self.trial_info['Image Onset NEV']
                fixation_onset = -trial_end
                if any(fixation_onset > 0) or any(fixation_onset < -5):
                    print(
                        f"trials with fixation_onset annotation problem {np.where((fixation_onset > 0) | (fixation_onset < -5))[0]}")
                    trial_end[(fixation_onset > 0) | (fixation_onset < -5)] = -1
                if any(trial_end < 0.1) or any(trial_end > 5):
                    print(
                        f"trials with trial_end annotation problem {np.where((trial_end < 0.1) | (trial_end > 5))[0]}")
                    trial_end[(trial_end < 0.1) | (trial_end > 5)] = 1
            elif settings.patient in ['p12', 'p13', 'p15', 'p16', 'p17', 'p18']:
                self.trial_info['Image Offset NEV'][:-1] = self.trial_info['Image Offset NEV'][1:]
                fixation_onset = self.trial_info['Fixation Onset NEV'] - self.trial_info['Image Onset NEV']
                trial_end = self.trial_info['Image Offset NEV'] - self.trial_info['Image Onset NEV']
                if any(fixation_onset > 0) or any(fixation_onset < -5):
                    print(
                        f"trials with fixation_onset annotation problem {np.where((fixation_onset > 0) | (fixation_onset < -5))[0]}")
                    trial_end[(fixation_onset > 0) | (fixation_onset < -5)] = -1
                if any(trial_end < 0.1) or any(trial_end > 5):
                    print(
                        f"trials with trial_end annotation problem {np.where((trial_end < 0.1) | (trial_end > 5))[0]}")
                    trial_end[(trial_end < 0.1) | (trial_end > 5)] = 1
            else:
                fixation_onset = self.trial_info['Fixation Onset NEV'] - self.trial_info['Image Onset NEV']
                trial_end = self.trial_info['Image Offset NEV'] - self.trial_info['Image Onset NEV']

            if any(trial_end > 5):
                print(f"trials with annotation problem {np.where(trial_end > 5)[0]}")
                trial_end[trial_end > 5] = 1

            self.time_annotation = {
                "fixation onset": np.round(np.mean(fixation_onset) * 1000),
                "image onset": 0,
                "trial end": np.round(np.mean(trial_end) * 1000)
            }

            self.trial_time_annotation = {
                "fixation onset": np.round(fixation_onset.values * 1000),
                "image onset": np.zeros(self.trial_info.shape[0]),
                "trial end": np.round(trial_end.values * 1000)
            }
        print(f"task {settings.task} : \n {self.time_annotation}")

    def _create_trial_indicator(self, task, label, trial_idx):
        time = self.time * 1000
        continuous_indicator = np.zeros_like(self.time) * np.nan

        if task == 'imagine':
            print()
            imagination_period_onset = np.argmin(np.abs(time))
            choice_period_onset = np.argmin(
                np.abs(time - self.trial_time_annotation['imagination_period_end'][trial_idx]))
            reaction_time_onset = np.argmin(np.abs(time - self.trial_time_annotation['reaction_time'][trial_idx]))
            image_onset = np.argmin(np.abs(time - self.trial_time_annotation['choice_period_end'][trial_idx]))
            trial_end = np.argmin(np.abs(time - self.trial_time_annotation['trial_end'][trial_idx]))

            continuous_indicator[imagination_period_onset:choice_period_onset] = 2
            continuous_indicator[choice_period_onset:image_onset] = 3
            if reaction_time_onset != 0:
                continuous_indicator[reaction_time_onset] = 4
            continuous_indicator[image_onset:trial_end] = 2 * label[trial_idx] - 1
        elif task == 'flicker_shape':
            image_onset = np.argmin(np.abs(time))
            fixation_onset = np.argmin(np.abs(time - self.trial_time_annotation['fixation onset'][trial_idx]))
            trial_end = np.argmin(np.abs(time - self.trial_time_annotation['trial end'][trial_idx]))

            continuous_indicator[fixation_onset:image_onset] = 0
            continuous_indicator[image_onset:trial_end] = 2 * label[trial_idx, 0] - 1
        elif task == 'flicker':
            image_onset = np.argmin(np.abs(time - self.trial_time_annotation['image onset'][trial_idx]))
            fixation_onset = np.argmin(np.abs(time - self.trial_time_annotation['fixation onset'][trial_idx]))
            trial_end = np.argmin(np.abs(time - self.trial_time_annotation['trial end'][trial_idx]))

            continuous_indicator[fixation_onset:image_onset] = 0
            continuous_indicator[image_onset:trial_end] = 2 * label[trial_idx] - 1
        else:
            raise ValueError("task not defined")

        return continuous_indicator

    def normalize_data(self, signal, moving_avg_size=0):
        min_val = np.quantile(signal, 0.025, axis=-1, keepdims=True)
        max_val = np.quantile(signal, 0.975, axis=-1, keepdims=True)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        normalized_signal[normalized_signal > 1] = 1
        normalized_signal[normalized_signal < 0] = 0

        if moving_avg_size > 0:
            window = np.ones(moving_avg_size) / moving_avg_size
            normalized_signal = np.apply_along_axis(non_overlapping_moving_average_repeated, 1, normalized_signal,
                                                    moving_avg_size)
        return normalized_signal

    @staticmethod
    def fir_bandpass(lowcut, highcut, fs, numtaps=101):
        nyq = 0.5 * fs
        taps = firwin(numtaps, [lowcut / nyq, highcut / nyq], pass_zero=False)
        return taps

    def get_continuous_features(self, channel_number, settings, save_path=None, debug=False, normalize_signal=True):
        if self.continuous_signal is None:
            self.epoched_to_continuous(settings, debug=debug, save_path=save_path)

        # Downsample continuous_indicator
        downsample_factor_2 = int(self.fs / 250)
        continuous_indicator_downsampled = self.continuous_indicator[::downsample_factor_2]
        time = np.arange(0, continuous_indicator_downsampled.shape[-1]) / 250

        # Filter for 0.3Hz to 20Hz
        # Design the FIR filter
        taps = self.fir_bandpass(0.3, 20, self.fs, numtaps=101)
        signal_1 = filtfilt(taps, [1.0], self.continuous_signal, axis=1)

        # plot_freq_response_fir(taps, fs=self.fs)

        # Downsample signal_1 to 250Hz
        downsample_factor_1 = int(self.fs / 250)
        signal_erp_downsampled = signal_1[:, ::downsample_factor_1]

        if normalize_signal is True:
            signal_erp_downsampled = self.normalize_data(signal=signal_erp_downsampled, moving_avg_size=0)
            signal_erp_downsampled = 2 * signal_erp_downsampled - 1

        """
        plot_continuous_heatmap(time=time,
                                signal=signal_erp_downsampled,
                                indicator=continuous_indicator_downsampled,
                                channel_names=self.channel_name,
                                normalize=normalize_signal,
                                cmap='jet',
                                save_path=save_path,
                                file_name='continuous_erp_heatmap')
        plot_continuous_signal(continuous_signal=signal_erp_downsampled,
                               continuous_indicator=continuous_indicator_downsampled,
                               channel_names=self.channel_name,
                               channel_number=channel_number,
                               settings=settings,
                               moving_avg_win=0,
                               save_path=save_path,
                               file_name='continuous_erp_signal')
                               """

        # Filter for 65Hz to 115Hz
        taps = self.fir_bandpass(65, 119, self.fs, numtaps=120)
        signal_2 = filtfilt(taps, [1.0], self.continuous_signal, axis=1)

        # plot_multitaper_psd(self.continuous_signal[0], fs=self.fs, NW=10)
        # plot_multitaper_psd(signal_2[0], fs=self.fs, NW=10)

        # Apply Hilbert transform
        signal_2_hilbert = np.abs(hilbert(signal_2, axis=1))

        # Downsample signal_2_hilbert to 250Hz
        downsample_factor_2 = int(self.fs / 250)
        signal_hga_downsampled = signal_2_hilbert[:, ::downsample_factor_2]

        # plot_multitaper_psd(signal_2_hilbert[0], fs=self.fs, NW=10)

        if normalize_signal is True:
            signal_hga_downsampled = self.normalize_data(signal=signal_hga_downsampled, moving_avg_size=0)
            signal_erp_downsampled = 2 * signal_erp_downsampled - 1

        """
        plot_continuous_heatmap(time=time,
                                signal=signal_hga_downsampled,
                                indicator=continuous_indicator_downsampled,
                                channel_names=self.channel_name,
                                cmap='jet',
                                normalize=normalize_signal,
                                save_path=save_path,
                                file_name='continuous_hga_heatmap')
        plot_continuous_signal(continuous_signal=signal_hga_downsampled,
                               continuous_indicator=continuous_indicator_downsampled,
                               channel_names=self.channel_name,
                               channel_number=channel_number,
                               settings=settings,
                               moving_avg_win=50,
                               save_path=save_path,
                               file_name='continuous_hga_signal')
        """

        return time, signal_erp_downsampled, signal_hga_downsampled, continuous_indicator_downsampled

    def get_trial_features(self, normalize_signal=True):
        # Downsample signal_1 to 250Hz
        downsample_factor = int(self.fs / 250)
        time_downsampled = self.time[::downsample_factor]

        # Filter for 0.3Hz to 20Hz
        # b, a = self.butter_bandpass(0.4, 20)
        taps = self.fir_bandpass(1, 20, self.fs, numtaps=200)
        erp_feature = np.zeros((self.data.shape[0], self.data.shape[1], len(time_downsampled)))

        for i in tqdm(range(self.data.shape[0])):
            # signal_1 = filtfilt(b, a, self.data[i, :, :], axis=1)
            signal_1 = filtfilt(taps, [1.0], self.data[i, :, :], axis=1)

            signal_1_downsampled = signal_1[:, ::downsample_factor]
            if normalize_signal is True:
                signal_1_downsampled = self.normalize_data(signal=signal_1_downsampled, moving_avg_size=0)
                signal_1_downsampled = 2 * signal_1_downsampled - 1
            erp_feature[i, :, :] = signal_1_downsampled

        # Filter for 65Hz to 115Hz
        hga_feature = np.zeros((self.data.shape[0], self.data.shape[1], len(time_downsampled)))
        # b, a = self.butter_bandpass(65, 115)
        # b2, a2 = self.butter_bandpass(0.4, 100)
        taps_ha1 = self.fir_bandpass(65, 119, self.fs, numtaps=120)
        taps_ha2 = self.fir_bandpass(65, 119, self.fs, numtaps=120)

        for i in tqdm(range(self.data.shape[0])):
            # signal_2 = filtfilt(b, a, self.data[i, :, :], axis=1)
            signal_2 = filtfilt(taps_ha1, [1.0], self.data[i, :, :], axis=1)
            # Apply Hilbert transform
            signal_2_hilbert = np.abs(hilbert(signal_2, axis=1))
            # Anti aliasing Filter
            signal_2_hilbert = filtfilt(taps_ha2, [1.0], signal_2_hilbert, axis=1)
            # downsample
            signal_hga = signal_2_hilbert[:, ::downsample_factor]

            if normalize_signal is True:
                signal_hga = self.normalize_data(signal=signal_hga, moving_avg_size=0)
                signal_hga = 2 * signal_hga - 1

            hga_feature[i, :, :] = signal_hga
        return time_downsampled, erp_feature, hga_feature

    def butter_bandpass(self, lowcut, highcut, order=4):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def plot_all_continuous_signal(self, settings, save_path=None, debug=False):
        if self.continuous_signal is None:
            self.epoched_to_continuous(settings, debug=debug, save_path=save_path)
        # Create the plot
        fig, ax = plt.subplots(figsize=(60, 327), dpi=200)
        offset = 0
        for channel_number in range(self.continuous_signal.shape[0] - 1):
            ax.plot(offset + self.continuous_signal[channel_number, :], 'k')  # 'k' stands for black color
            offset += np.quantile(self.continuous_signal[channel_number, :], 0.99) - \
                      np.quantile(self.continuous_signal[channel_number + 1, :], 0.01) * \
                      (np.quantile(self.continuous_signal[channel_number + 1, :], 0.01) < 0)
        # Find start and end indices for -1 regions
        neg_starts = np.where(np.diff(np.hstack(([0], self.continuous_indicator == -1, [0]))) == 1)[0]
        neg_ends = np.where(np.diff(np.hstack(([0], self.continuous_indicator == -1, [0]))) == -1)[0]

        # Find start and end indices for 1 regions
        pos_starts = np.where(np.diff(np.hstack(([0], self.continuous_indicator == 1, [0]))) == 1)[0]
        pos_ends = np.where(np.diff(np.hstack(([0], self.continuous_indicator == 1, [0]))) == -1)[0]

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
            imagination_starts = np.where(np.diff(np.hstack(([0], self.continuous_indicator == 2, [0]))) == 1)[0]
            imagination_ends = np.where(np.diff(np.hstack(([0], self.continuous_indicator == 2, [0]))) == -1)[0]

            # Find start and end indices for 1 regions
            choice_starts = np.where(np.diff(np.hstack(([0], self.continuous_indicator == 3, [0]))) == 1)[0]
            choice_ends = np.where(np.diff(np.hstack(([0], self.continuous_indicator == 3, [0]))) == -1)[0]

            selection_starts = np.where(np.diff(np.hstack(([0], self.continuous_indicator == 4, [0]))) == 1)[0]
            selection_ends = np.where(np.diff(np.hstack(([0], self.continuous_indicator == 4, [0]))) == -1)[0]

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

        ax.set_title(self.channel_name[channel_number])
        ax.legend()
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else:
            fig.savefig(save_path + f"continuous_signals_all.png")

    def extreact_continuous_data_from_epochs(self):
        num_trial, num_channel, num_sample = self.data.shape
        if self.trial_info is None:
            raise ValueError("To make continuous data from epoched data you need trial_info but it's None")

        trial_list = []
        event_indicator_list = []
        for i in self.trial_info['Trial number'].to_list():
            row = self.trial_info.loc[self.trial_info['Trial number'] == i, :]
            idx = row.index[0]
            # get the time interval from fixation to next fixateio
            time_range = [-row['fixation_time'].values[0], row['display_time'].values[0]]

            # find time idx
            t_start = np.argmin(np.abs(self.time - time_range[0]))
            t_end = np.argmin(np.abs(self.time - time_range[1]))
            t_origin = np.argmin(np.abs(self.time - 0))

            # get label from data info
            label = row['ColorLev'].values[0]

            # crate indicator signal
            event_indicator = np.zeros(self.time.shape)
            event_indicator[int(len(self.time) / 2):] = 1 if label == 1 else -1

            # truncated time
            trial_list.append(self.data[idx, :, t_start:t_end])
            event_indicator_list.append(event_indicator[t_start:t_end])

        continuous_data = np.concatenate(trial_list, axis=1)
        event_indicator = np.concatenate(event_indicator_list, axis=0)
        time = np.arange(0, len(event_indicator)) / self.fs

        return time, continuous_data, event_indicator

    def get_epochs_from_continuous_data(self, continuous_data):
        num_trial, num_channel, num_sample = self.data.shape
        epoched_data = np.zeros_like(self.data)
        if self.trial_info is None:
            raise ValueError("To make continuous data from epoched data you need trial_info but it's None")

        trial_list = []
        event_indicator_list = []
        for i in range(1, num_trial + 1):
            row = self.trial_info.loc[self.trial_info['Trial number'] == i, :]
            idx = row.index[0]
            # get the time interval from fixation to next fixation
            time_range = [-row['fixation_time'].values[0], row['display_time'].values[0]]

            # find time idx
            t_start = np.argmin(np.abs(self.time - time_range[0]))
            t_end = np.argmin(np.abs(self.time - time_range[1]))
            t_origin = np.argmin(np.abs(self.time - 0))

            epoched_data

            # get label from data info
            label = row['ColorLev'].values[0]

            # crate indicator signal
            event_indicator = np.zeros(self.time.shape)
            event_indicator[5000:] = 1 if label == 1 else -1

            # truncated time
            trial_list.append(self.data[idx, :, t_start:t_end])
            event_indicator_list.append(event_indicator[t_start:t_end])

        continuous_data = np.concatenate(trial_list, axis=1)
        event_indicator = np.concatenate(event_indicator_list, axis=0)
        time = np.arange(0, len(event_indicator)) / self.fs
