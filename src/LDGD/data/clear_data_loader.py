import numpy as np
import pandas as pd
from scipy import io
import h5py
from .data_loader import IEEGData

class IEEGDataLoader:
    def __init__(self, paths, settings):
        self.target_class = settings.target_class
        self.patient = settings.patient
        self.experiment = settings.experiment
        self.task = 'FlickerShapes'  # 'FlickerShapes' or 'Flicker'
        self.paths = paths
        self.results = {}

    def load_data(self):

        data_matlab_class1_hgp = io.loadmat(self.paths.path_dataset_hgp[0])
        data_matlab_class2_hgp = io.loadmat(self.paths.path_dataset_hgp[1])
        data_matlab_class1_erp = io.loadmat(self.paths.path_dataset_erp[0])
        data_matlab_class2_erp = io.loadmat(self.paths.path_dataset_erp[1])

        if self.target_class == 'color':
            data_matlab_class1_hgp = data_matlab_class1_hgp['ft_FrerqData_Black_filtered'][0]
            data_matlab_class2_hgp = data_matlab_class2_hgp['ft_FrerqData_White_filtered'][0]
            data_matlab_class1_erp = data_matlab_class1_erp['ft_data_Black_lp_ds'][0]
            data_matlab_class2_erp = data_matlab_class2_erp['ft_data_White_lp_ds'][0]
        elif self.target_class == 'shape':
            data_matlab_class1_hgp = data_matlab_class1_hgp['ft_FrerqData_Shape1_filtered'][0]
            data_matlab_class2_hgp = data_matlab_class2_hgp['ft_FrerqData_Shape2_filtered'][0]
            data_matlab_class1_erp = data_matlab_class1_erp['ft_data_Shape1_lp_ds'][0]
            data_matlab_class2_erp = data_matlab_class2_erp['ft_data_Shape2_lp_ds'][0]
        elif self.target_class == 'tone':
            data_matlab_class1_hgp = data_matlab_class1_hgp['ft_FrerqData_Tone1_filtered'][0]
            data_matlab_class2_hgp = data_matlab_class2_hgp['ft_FrerqData_Tone2_filtered'][0]
            data_matlab_class1_erp = data_matlab_class1_erp['ft_data_Tone1_lp_ds'][0]
            data_matlab_class2_erp = data_matlab_class2_erp['ft_data_Tone2_lp_ds'][0]
        else:
            raise ValueError('target class ' + self.target_class + ' is not defined')

        time = np.squeeze(data_matlab_class1_hgp['time'][0][0][0])

        # specify data dimesions
        num_trials_class1 = data_matlab_class1_hgp['trial'][0][0].shape[0]
        num_trials_class2 = data_matlab_class2_hgp['trial'][0][0].shape[0]

        num_channels, num_samples_erp = data_matlab_class1_erp['trial'][0][0][0].shape
        num_channels, num_samples_hgp = data_matlab_class1_hgp['trial'][0][0][0].shape

        if num_samples_erp != num_samples_hgp:
            print('ERP number of samples is ' + str(num_samples_hgp) +
                  ' but HGP number of samples is ' + str(num_samples_hgp))
            num_samples = np.min([num_samples_hgp, num_samples_hgp])
        else:
            num_samples = num_samples_erp

        time = time[:num_samples]

        # change data into numpy array
        data_erp_class1 = np.zeros((num_trials_class1, num_channels, num_samples))
        data_hgp_class1 = np.zeros((num_trials_class1, num_channels, num_samples))
        for i in range(num_trials_class1):
            data_erp_class1[i, :, :] = data_matlab_class1_erp['trial'][0][0][i][:, :num_samples]
            data_hgp_class1[i, :, :] = data_matlab_class1_hgp['trial'][0][0][i][:, :num_samples]

        data_erp_class2 = np.zeros((num_trials_class2, num_channels, num_samples))
        data_hgp_class2 = np.zeros((num_trials_class2, num_channels, num_samples))
        for i in range(num_trials_class2):
            data_erp_class2[i, :, :] = data_matlab_class2_erp['trial'][0][0][i][:, :num_samples]
            data_hgp_class2[i, :, :] = data_matlab_class2_hgp['trial'][0][0][i][:, :num_samples]

        # channel names
        channel_names = data_matlab_class1_erp['ChannelPairNamesBankAll'][0].tolist()
        channel_names_erp = [channel_names[i] + ' ERP' for i in range(len(channel_names))]
        channel_names_hgp = [channel_names[i] + ' HGP' for i in range(len(channel_names))]

        self.results = {
            'data_erp1': data_erp_class1,
            'data_erp2': data_erp_class2,
            'data_hgp1': data_hgp_class1,
            'data_hgp2': data_hgp_class1,
            'channel_names_erp': channel_names_erp,
            'channel_names_hgp': channel_names_hgp,
            'time': time,
            'fs': 15
        }

        return self.results

    def load_data_combined_class(self):
        if self.results == {}:
            self.load_data()

        labels = np.concatenate(
            [np.zeros((self.results['data_erp1'].shape[0],)),
             np.ones((self.results['data_erp2'].shape[0],))],
            axis=0)

        data_erp = np.concatenate(
            [self.results['data_erp1'], self.results['data_erp2']],
            axis=0)

        data_hgp = np.concatenate(
            [self.results['data_hgp1'], self.results['data_hgp2']],
            axis=0)

        self.results = {
            'data_erp': data_erp,
            'data_hgp': data_hgp,
            'labels': labels,
            'channel_names_erp': self.results['channel_names_erp'],
            'channel_names_hgp': self.results['channel_names_hgp'],
            'time': self.results['time'],
            'fs': 15
        }

        return self.results

    def load_raw_data(self):
        print("Loading Raw Data ...")
        mat_file = io.loadmat(self.paths.path_dataset_raw[0])

        data = np.array(mat_file['ft_data'][0]['trial'][0])
        label = np.squeeze(np.array(mat_file['ft_data'][0]['label'][0]))
        channel_name = mat_file['ft_data'][0]['channel_name'][0]
        channel_name = [element[0] for element in channel_name[0].tolist()]
        fs = mat_file['ft_data'][0]['fsample'][0][0, 0]

        # trial info

        if self.patient.lower() == 'p05':
            trial_info = mat_file['ft_data'][0]['trialinfo'][0][:, 1:-4]
            trial_info = np.delete(trial_info, 10, axis=1)
            trial_info_hdr = mat_file['ft_data'][0]['trialinfohdr'][0]
            trial_info_hdr = [element[0] for element in trial_info_hdr[0].tolist()]
            trial_info_hdr.append('Experiment')
            trial_info = pd.DataFrame(trial_info, columns=trial_info_hdr)
            trial_info['fixation_time'] = trial_info['Image Onset NEV'] - trial_info['Fixation Onset NEV']
            trial_info['display_time'] = trial_info['TimeTrial'] - trial_info['TimeImageOnset']
        elif self.patient.lower() == 'p02':
            trial_info = mat_file['ft_data'][0]['trialinfo'][0][:, 1:21]
            trial_info = np.delete(trial_info, 14, axis=1)
            trial_info_hdr = mat_file['ft_data'][0]['trialinfohdr'][0]
            trial_info_hdr = [element[0] for element in trial_info_hdr[0].tolist()]
            trial_info_hdr.remove('TimeRec')
            trial_info_hdr = trial_info_hdr[:trial_info.shape[-1] - 1]
            trial_info_hdr.append('Experiment')

            trial_info = pd.DataFrame(trial_info, columns=trial_info_hdr)
            trial_info['fixation_time'] = trial_info['Image Onset NEV'] - trial_info['Fixation Onset NEV']
            trial_info['display_time'] = trial_info['TimeTrial'] - trial_info['TimeImageOnset']
            # trial_info.to_csv('p02.csv')
        elif self.patient.lower() in ['p01', 'p03']:
            trial_info = mat_file['ft_data'][0]['trialinfo'][0][:, 1:21]
            trial_info = np.delete(trial_info, 17, axis=1)
            trial_info_hdr = mat_file['ft_data'][0]['trialinfohdr'][0]
            trial_info_hdr = [element[0] for element in trial_info_hdr[0].tolist()]
            trial_info_hdr.remove('TimeRec')
            trial_info_hdr = trial_info_hdr[:trial_info.shape[-1] - 1]
            trial_info_hdr.append('Experiment')

            trial_info = pd.DataFrame(trial_info, columns=trial_info_hdr)
            trial_info['fixation_time'] = trial_info['Image Onset NEV'] - trial_info['Fixation Onset NEV']
            trial_info['display_time'] = trial_info['TimeTrial'] - trial_info['TimeImageOnset']
            # trial_info.to_csv('p01.csv')

        if self.experiment is not None:
            print("Number of trial in this experiment is " + str(np.sum(trial_info['Experiment'] == self.experiment)))
            data = data[trial_info['Experiment'].values == self.experiment]
            label = label[trial_info['Experiment'].values == self.experiment]
            trial_info = trial_info[trial_info['Experiment'] == self.experiment].reset_index(drop=True)

            sorted_index = trial_info.sort_values(by='Trial number', ascending=True).index
            trial_info = trial_info.reindex(sorted_index).reset_index(drop=True)

            label = label[sorted_index]
            data = data[sorted_index]
        else:
            print("Number of trial in all experiment are " + str(trial_info.shape[0]))
            for exp in list(np.unique(trial_info['Experiment'])):
                print(
                    "Number of trial in experiment " + str(exp) + " is " + str(np.sum(trial_info['Experiment'] == exp)))
            trial_info = trial_info.reset_index(drop=True)

            sorted_index = trial_info.sort_values(by='Trial number', ascending=True).index
            trial_info = trial_info.reindex(sorted_index).reset_index(drop=True)

            label = label[sorted_index]
            data = data[sorted_index]

        time = np.squeeze(np.array(mat_file['ft_data'][0]['time'][0][0, 0]))

        data_ieeg = IEEGData(data, fs)
        data_ieeg.time = time
        data_ieeg.channel_name = channel_name
        data_ieeg.trial_info = trial_info
        data_ieeg.label = label

        return data_ieeg

    def load_data_from_ft_data(self):
        print("Loading Raw Data ...")
        merged_bank_file = h5py.File(self.paths.path_dataset_mergedbank[0], 'r')
        field_trip_file = h5py.File(self.paths.path_dataset_fieldtrip[0], 'r')

        ft_data = field_trip_file['ft_data3']
        fs = np.squeeze(ft_data['fsample'][()])
        if len(merged_bank_file['ChannelPairNamesBank1'].shape) > 1:
            channel_names = ["".join(chr(char) for char in cell) for cell in
                             merged_bank_file['ChannelPairNamesBank1'][()].transpose()]
            if len(merged_bank_file['ChannelPairNamesBank2'].shape) > 1:
                channel_names = channel_names + ["".join(chr(char) for char in cell) for cell in
                                                 merged_bank_file['ChannelPairNamesBank2'][()].transpose()]
        elif len(merged_bank_file['ChannelPairNames'].shape) > 1:
            channel_names = ["".join(chr(char) for char in cell) for cell in
                             merged_bank_file['ChannelPairNames'][()].transpose()]
        else:
            channel_names = []
            print(" No Channel Name is available")

        data_list = [field_trip_file[ref[0]][()].transpose() for ref in field_trip_file['ft_data3']['trial'][()]]
        data = np.stack(data_list, axis=0)
        time = np.squeeze(field_trip_file[field_trip_file['ft_data3']['time'][()][0][0]][()])
        experiment_type = np.squeeze(merged_bank_file['TrialTypeDesignation'][()])
        trial_info_hdr = [np.squeeze(field_trip_file[ref[0]][()]) for ref in
                          field_trip_file['TrialAlignedMatHeaders'][()]]
        trial_info_hdr = ["".join(chr(char) for char in list(info)) for info in trial_info_hdr]
        if self.patient in ['p05', 'p06']:
            columns_name = ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'Stim on/off NEV',
                            'Stim onset NEV', 'Stim offset NEV', 'Response value NEV', 'Reaction onset NEV',
                            'Congruent Value NEV', 'Valence Value NEV', 'Reaction time NEV', 'Noname', 'TrialNum',
                            'Delay1', 'Delay2', 'TrialComplete', 'TimeImageOnset', 'ColorLev', 'TimeRec', 'TimeTrial',
                            'BlockNumber']
        elif self.patient in ['p01', 'p03', 'p04']:
            columns_name = ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'Stim on/off NEV',
                            'Stim onset NEV', 'Stim offset NEV', 'Response value NEV', 'Reaction onset NEV',
                            'Congruent Value NEV', 'Valence Value NEV', 'Reaction time NEV', 'TrialNum',
                            'Delay1', 'Delay2', 'TrialComplete', 'TimeImageOnset', 'ColorLev',
                            'TimeRec', 'TimeTrial', 'BlockNumber']
        elif self.patient in ['p02']:
            columns_name = ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'Stim on/off NEV',
                            'Stim onset NEV', 'Stim offset NEV', 'Response value NEV', 'Reaction onset NEV',
                            'Congruent Value NEV', 'Valence Value NEV', 'Reaction time NEV', 'TrialNum',
                            'Delay1', 'Delay2', 'TrialComplete', 'TimeImageOnset', 'Noname', 'ColorLev',
                            'TimeTrial', 'BlockNumber']
        elif self.patient in ['p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17',
                              'p18']:
            columns_name = ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'NoName', 'Stim on/off NEV',
                            'Stim onset NEV', 'Stim offset NEV',  # 'Response value NEV', 'Reaction onset NEV'
                            'Congruent Value NEV', 'Valence Value NEV', 'Reaction time NEV', 'NoName1', 'NoName2',
                            'TrialNum',
                            'Delay1', 'Delay2', 'TrialComplete', 'TimeImageOnset', 'ColorLev',
                            'TimeRec', 'TimeTrial', 'StartTrial', 'CrossStart', 'ResponseTime']
            if 'ShapeChoice' in trial_info_hdr:
                columns_name.append('ShapeChoice')
            columns_name.append('BlockNumber')
        else:
            raise ValueError("No Trial infor alignment is specified for {}".format(self.patient))

        trail_info = field_trip_file['TrialAlignedMat1'][()].transpose()
        if trail_info.shape[1] > len(columns_name):
            columns_name = columns_name + ['noname'] * (trail_info.shape[1] - len(columns_name))
        else:
            columns_name = columns_name[:trail_info.shape[1]]
        trial_info = pd.DataFrame(trail_info, columns=columns_name)
        trial_info['TrialTypeDesignation'] = experiment_type
        trial_info.to_csv(self.patient + '.csv', index=False)

        df2 = trial_info.dropna(axis=1, how='all')
        """ p01 ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'Stim on/off NEV', 'Stim onset NEV', 
        'Stim offset NEV', 'Response value NEV', 'TrialNum', 'Delay1', 'Delay2', 
        'TrialComplete', 'TimeImageOnset', 'ColorLev', 'TimeRec', 'TimeTrial', 'BlockNumber', 'TrialTypeDesignation']"""

        """ p02 ['Trial number', 'Fixation Onset NEV', 'Image Onset NEV', 'Stim on/off NEV', 'Stim onset NEV', 
        'Stim offset NEV', 'Response value NEV', 'Valence Value NEV', 'TrialNum', 'Delay1', 'Delay2', 
        'TrialComplete', 'ColorLev', 'TimeTrial', 'TrialTypeDesignation']
       """
