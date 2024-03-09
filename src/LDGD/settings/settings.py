import json
import os
import warnings


class Settings:
    def __init__(self, target_class='color', task='flicker', model='btsc', patient='p05',
                 classifier_name='bayesian_model', combination_mode='maxvote', feature_type='combine', verbose=True):

        self.verbose = verbose
        self.__supported_feature_types = ['combine', 'hgp', 'erp']
        self.__supported_classifiers = ['bayesian_model',
                                        'svc', 'naive_bayes', 'logistic_regression', 'random_forest', 'mlp', 'lstm']
        self.__supported_combination_mode = ['pol', 'maxvote']
        self.__supported_experiments = {'p01': 1, 'p02': 1, 'p03': 4, 'p04': 99, 'p05': 2, 'p06': 99, 'p07': 99,
                                        'p08': 99, 'p09': 99, 'p10': 99, 'p11': 99, 'p12': 99, 'p13': 99, 'p14': 99,
                                        'p15': 99, 'p16': 99, 'p17': 99, 'p18': 99}
        self.__supported_dataset = ['oil', 'wine', 'iris', 'usps', 'synthetic', 'mnist']
        self.__supported_models = ['jbgplvm']
        self.__supported_task = ['m_sequence', 'flicker', 'flicker_shape', 'imagine']
        self.__model = model
        self.__patient = patient
        self.__task = task
        self.__classifier_name = classifier_name  # svc, naive_bayes, logistic_regression, random_forest, mlp, lstm
        self.__combination_mode = combination_mode
        self.__feature_type = feature_type
        self.__experiment = self.__supported_experiments[patient]
        self.__debug_mode = False
        self.__load_feature = False
        self.__save_features = False
        self.__load_pretrained_model = False
        self.__num_fold = 5
        self.__test_size = 0.2
        self.__file_format = '.pkl'
        self.__dataset = self.__supported_dataset[0]
        self.__model = None

        if not isinstance(target_class, str):
            raise ValueError('"target_class" must be string!')
        else:
            if target_class.lower() not in ['color', 'shape']:
                raise ValueError('"target_class" must be color or shape!')
            self.__target_class = target_class.lower()

    def load_settings(self):
        """
        This function loads the json files for settings and network settings from the working directory and
        creates a Settings object based on the fields in the json file. It also loads the local path of the dataset
        from device_path.json
        return:
            settings: a Settings object
            network_settings: a dictionary containing settings of the model
            device_path: the path to the datasets on the local device
        """

        """ working directory """
        working_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        parent_folder = os.path.dirname(os.path.dirname(working_folder))+'\\'

        """ loading settings from the json file """
        try:
            with open(parent_folder + "/configs/settings.json", "r") as file:
                settings_json = json.loads(file.read())
        except:
            raise Exception('Could not load settings.json from the working directory!')

        """ creating settings """
        if "dataset" not in settings_json.keys():
            raise Exception('"dataset" was not found in settings.json!')
        else:
            self.dataset = settings_json["dataset"]

        if "model" not in settings_json.keys():
            raise Exception('"model" was not found in settings.json!')
        else:
            self.model = settings_json["model"]
            del settings_json["model"]

        if self.dataset.lower() == 'ieeg':
            if "target_class" not in settings_json.keys():
                raise Exception('"target_class" was not found in settings.json!')
            if "task" not in settings_json.keys():
                raise Exception('"task" was not found in settings.json!')
            if "patient" not in settings_json.keys():
                raise Exception('"patient" was not found in settings.json!')



            self.target_class = settings_json["target_class"]
            self.task = settings_json["task"]
            self.patient = settings_json["patient"]


            del settings_json["target_class"]
            del settings_json["task"]
            del settings_json["patient"]


        for key, value in settings_json.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise Exception('{} is not an attribute of the Settings class!'.format(key))
        if self.verbose is True and self.dataset.lower() == 'ieeg':
            print("Patient: {} Feature type: {} Combination method: {}".format(self.patient,
                                                                               self.feature_type,
                                                                               self.combination_mode))

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset_name):
        if isinstance(dataset_name, str) and dataset_name in self.__supported_dataset:
            print(f"\n Selected Dataset is : {dataset_name}")
            self.__dataset = dataset_name
        else:
            raise ValueError(f"dataset should be selected from supported datasets: {self.__supported_dataset}")

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model_name):
        if isinstance(model_name, str) and model_name in self.__supported_models:
            print(f"\n Selected Dataset is : {model_name}")
            self.__model = model_name
        else:
            raise ValueError(f"dataset should be selected from supported datasets: {self.__supported_models}")


    @property
    def file_format(self):
        return self.__file_format

    @file_format.setter
    def file_format(self, format_name):
        if isinstance(format_name, str) and format_name[0]=='.':
            self.__file_format = format_name
        else:
            raise ValueError(f"file_format should be a string starting with . (.pkl, .mat, and etc.) "
                             f"but we got {format_name}")
    @property
    def num_fold(self):
        return self.__num_fold

    @num_fold.setter
    def num_fold(self, k):
        if isinstance(k, int) and k > 0:
            self.__num_fold = k
        else:
            raise ValueError("num_fold should be integer bigger than 0")

    @property
    def test_size(self):
        return self.__test_size

    @test_size.setter
    def test_size(self, value):
        if 0 < value < 1:
            self.__test_size = value
        else:
            raise ValueError("test_size should be float number between 0 to 1")

    @property
    def load_pretrained_model(self):
        return self.__load_pretrained_model

    @load_pretrained_model.setter
    def load_pretrained_model(self, value):
        if isinstance(value, bool):
            self.__load_pretrained_model = value
        else:
            raise ValueError("load_pretrained_model should be True or False")

    @property
    def load_feature(self):
        return self.__load_feature

    @load_feature.setter
    def load_feature(self, value):
        if isinstance(value, bool):
            self.__load_feature = value
        else:
            raise ValueError("load_feature should be True or False")

    @property
    def save_features(self):
        return self.__save_features

    @save_features.setter
    def save_features(self, value):
        if isinstance(value, bool):
            self.__save_features = value
        else:
            raise ValueError("save_features should be True or False")

    @property
    def debug_mode(self):
        return self.__debug_mode

    @debug_mode.setter
    def debug_mode(self, value):
        if isinstance(value, bool):
            self.__debug_mode = value
        else:
            raise ValueError("The v should be boolean")

    @property
    def experiment(self):
        return self.__experiment

    @experiment.setter
    def experiment(self, experimente_mode):
        if experimente_mode.lower() == 'all':
            self.__experiment = None
        elif experimente_mode.lower() == 'auto':
            if self.patient in self.__supported_experiments:
                self.__experiment = self.__supported_experiments[self.patient]
            else:
                self.__experiment = None

    @property
    def feature_type(self):
        return self.__feature_type

    @feature_type.setter
    def feature_type(self, f_type):
        if f_type in self.__supported_feature_types:
            self.__feature_type = f_type
        else:
            raise ValueError("Feature type {} is not supporter. It should be selected from {}".format(
                f_type, self.__supported_feature_types
            ))

    @property
    def combination_mode(self):
        return self.__combination_mode

    @combination_mode.setter
    def combination_mode(self, mode):
        if mode.lower() in self.__supported_combination_mode:
            self.__combination_mode = mode
        else:
            raise ValueError("Combination mode {} is not in supported methods. Choose from {}".format(
                mode, self.__supported_combination_mode))

    @property
    def classifier_name(self):
        return self.__classifier_name

    @classifier_name.setter
    def classifier_name(self, name):
        if name.lower() in self.__supported_classifiers:
            self.__classifier_name = name
        else:
            raise ValueError(
                "The classifier {} is not suppoerted. Choose from {}".format(name, self.__supported_classifiers))

    @property
    def target_class(self):
        return self.__target_class
    @target_class.setter
    def target_class(self, target):
        self.__target_class = target

    @property
    def patient(self):
        return self.__patient

    @patient.setter
    def patient(self, patient_list):
        self.__patient = patient_list

    @property
    def data_labels(self):
        return self.__data_labels

    @property
    def task(self):
        return self.__task

    @task.setter
    def task(self, task_name):
        if task_name in self.__supported_task:
            self.__task = task_name
        else:
            self.__task = task_name
            print(f'Warning No task specified with name {task_name} please use one of supported tasks: '
                  f'{self.__supported_task}. Task is set to {task_name}')

