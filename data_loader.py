import csv
import numpy as np
import torch


class DataLoader:
    def __init__(self, x_seq_length, x_features, y_seq_length, mode='train', return_format='keras',
                 torch_device=None):
        self.x_seq_length = x_seq_length
        self.x_features = x_features
        self.y_seq_length = y_seq_length
        self.return_format = return_format
        self.torch_device = torch_device
        self.train_x_set = []
        self.train_y_set = []
        self.val_x_set = []
        self.val_y_set = []
        self.pred_x_set = []
        self.prepare_data(mode)

    @staticmethod
    def load_csv_data(filename_x, filename_y=None):
        with open(filename_x, 'r') as file1:
            reader1 = csv.reader(file1, quoting=csv.QUOTE_NONNUMERIC)
            if filename_y is None:
                return [row for row in reader1]
            with open(filename_y, 'r') as file2:
                reader2 = csv.reader(file2, quoting=csv.QUOTE_NONNUMERIC)
                return [row for row in reader1], [row for row in reader2]

    def format_data(self, s_x, s_y):
        if self.return_format == 'torch':
            s_x = torch.from_numpy(s_x)
            s_y = torch.from_numpy(s_y)
            if self.torch_device is not None:
                return s_x.to(device=self.torch_device), s_y.to(device=self.torch_device)
        return s_x, s_y

    def get_batch(self, n=20, source='train'):
        x_set, y_set, size = self.get_set_data(source)
        rng = np.random.default_rng()
        indices = rng.integers(0, size, n)
        s_x = np.take(a=x_set, indices=indices, axis=0).astype('f4').reshape((n, self.x_seq_length, self.x_features))
        s_y = np.take(a=y_set, indices=indices, axis=0).astype('f4')
        return self.format_data(s_x, s_y)

    def get_all_data(self, source='train'):
        x_set, y_set, size = self.get_set_data(source)
        s_x = np.array(x_set, dtype='f4').reshape((size, self.x_seq_length, self.x_features))
        s_y = np.array(y_set, dtype='f4')
        return self.format_data(s_x, s_y)

    def get_set_data(self, source='train'):
        assert source in ['train', 'validate', 'predict']
        if source == 'train':
            x_set = self.train_x_set
            y_set = self.train_y_set
        elif source == 'validate':
            x_set = self.val_x_set
            y_set = self.val_y_set
        else:
            x_set = self.pred_x_set
            y_set = []
        size = len(x_set)
        return x_set, y_set, size

    def prepare_data(self, mode='train'):
        # Use hardcoded filenames.
        filename_train_x = f'data/train_x_{self.x_seq_length}_{self.x_features}.csv'
        filename_train_y = f'data/train_y_{self.y_seq_length}.csv'
        filename_val_x = f'data/val_x_{self.x_seq_length}_{self.x_features}.csv'
        filename_val_y = f'data/val_y_{self.y_seq_length}.csv'
        filename_pred_x = f'data/pred_x_{self.x_seq_length}_{self.x_features}.csv'

        if mode == 'train' and len(self.train_y_set) == 0:
            print('Loading training data.')
            self.train_x_set, self.train_y_set = self.load_csv_data(filename_train_x, filename_train_y)
            n_samples = len(self.train_y_set)
            print(f'Training samples: {n_samples}')
            print(f'Sample x: {self.train_x_set[0]}')
            print(f'Sample y: {self.train_y_set[0]}')

        if mode in ['train', 'validate'] and len(self.val_y_set) == 0:
            print('Loading validation data.')
            self.val_x_set, self.val_y_set = self.load_csv_data(filename_val_x, filename_val_y)
            n_samples = len(self.val_y_set)
            print(f'Validation samples: {n_samples}')

        if mode == 'predict' and len(self.pred_x_set) == 0:
            print('Loading prediction data.')
            self.pred_x_set = self.load_csv_data(filename_pred_x)
            n_samples = len(self.pred_x_set)
            print(f'Prediction samples: {n_samples}')
