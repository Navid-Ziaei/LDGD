import json
import datetime
from pathlib import Path
import os


class Paths:
    def __init__(self, settings):

        self.class_label_list = None
        self.folder_name = None
        self.base_path = None

        self.dataset = settings.dataset
        self.combination_mode = settings.combination_mode
        self.feature_type = settings.feature_type
        self.task = settings.task
        self.target_class = settings.target_class
        self.patient = settings.patient
        self.debug_mode = settings.debug_mode
        self.num_folds = settings.num_fold

        self.raw_dataset_path = None
        self.prepared_dataset_path = None
        self.preprocessed_dataset_path = None

        self.path_dataset_raw = []
        self.path_model = []
        self.path_result = []

        self.raw_dataset = './'
        self.feature_path = '/'
        self.model = './'

    def load_device_paths(self):

        """ working directory """
        working_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        working_folder = os.path.dirname(os.path.dirname(working_folder))

        """ loading device path from the json file """
        try:
            with open(working_folder + "/configs/device_path.json", "r") as file:
                device = json.loads(file.read())
        except:
            raise Exception('Could not load device_path.json from the working directory!')

        for key, value in device.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception('{} is not an attribute of the Settings class!'.format(key))

        self.create_paths()

    def create_paths(self):
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # get the working directory path
        dir_path = os.path.dirname(os.path.dirname(dir_path))

        self.base_path = dir_path + '\\results'
        results_base_path = f'{self.base_path}\\{self.dataset}\\'

        if self.debug_mode is False:
            self.folder_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.folder_name = 'debug'
        results_base_path = results_base_path + self.folder_name + '\\'
        """if Path(results_base_path).is_dir():
            shutil.rmtree(results_base_path)"""

        model_path = os.path.join(results_base_path + 'model\\')
        Path(results_base_path).mkdir(parents=True, exist_ok=True)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.path_model.append(model_path)
        self.path_result.append(results_base_path)


    def create_paths_ieeg(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))  # get the working directory path
        dir_path = os.path.dirname(os.path.dirname(dir_path))

        self.base_path = dir_path + '\\results\\'
        results_base_path = self.base_path + '\\{}\\{}\\'.format(self.task, self.patient) + \
                            self.patient + '_' + self.feature_type + '_' + self.combination_mode + '\\'
        if self.debug_mode is False:
            self.folder_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.folder_name = 'debug'
        results_base_path = results_base_path + '/' + self.folder_name + '/'
        """if Path(results_base_path).is_dir():
            shutil.rmtree(results_base_path)"""

        Path(results_base_path).mkdir(parents=True, exist_ok=True)
        self.path_model.append(os.path.join(results_base_path + '/model/'))
        self.path_result.append(os.path.join(results_base_path + '/'))

        if self.num_folds > 1:
            for i in range(self.num_folds):
                Path(results_base_path + '/fold{}/model'.format(i + 1)).mkdir(parents=True, exist_ok=True)
                Path(results_base_path + '/fold{}'.format(i + 1)).mkdir(parents=True, exist_ok=True)
                self.path_model.append(os.path.join(results_base_path + '/fold{}/model/'.format(i + 1)))
                self.path_result.append(os.path.join(results_base_path + '/fold{}/'.format(i + 1)))

        if self.target_class == 'color':
            self.class_label_list = ['Black', 'White']
        elif self.target_class == 'shape':
            self.class_label_list = ['Shape1', 'Shape2']
        elif self.target_class == 'tone':
            self.class_label_list = ['Tone1', 'Tone2']
        elif isinstance(self.target_class, list):
            self.class_label_list = self.target_class
        else:
            raise ValueError('The target ' + self.target_class + ' is not defined')

        self.path_dataset_raw.append(self.raw_dataset + '/Processed/Data_' + self.task + '/' + self.patient +
                                     'data_single_file.mat')
    def update_path(self, time_index):
        self.path_model_updated, self.path_result_updated = [], []
        for idx in range(len(self.path_result)):
            self.path_model_updated.append(self.path_result[idx] + '/t_{}/model/'.format(time_index))
            self.path_result_updated.append(self.path_result[idx] + '/t_{}/'.format(time_index))
            Path(self.path_model_updated[idx]).mkdir(parents=True, exist_ok=True)
            Path(self.path_result_updated[idx]).mkdir(parents=True, exist_ok=True)
