from torch.utils.data import Dataset, DataLoader
import numpy as np

class healthDataset(Dataset):
    def __init__(self, np_features, np_labels, pid_list):
        self.np_features = np_features
        self.np_labels = np_labels
        self.pid_list = pid_list

        bool_array = np.sum([self.np_features[:,0] == pid for pid in self.pid_list], axis=0).astype('bool')
        self.np_features_filtered = self.np_features[bool_array, :]

    def __len__(self):
        return int(self.np_features_filtered.shape[0]/12)

    def __getitem__(self, index):
        """Generates one sample of data"""
        pid = self.np_features_filtered[index*12,0]

        features = self.np_features_filtered[index*12:index*12+12,3:]
        stacked_features = np.reshape(features, -1, order='F')

        bool_array = np.array([self.np_labels[:, 0] == pid])[0].astype('bool')
        labels = self.np_labels[bool_array, 1:12]
        stacked_labels = np.reshape(labels, -1, order='F')

        return (stacked_features, stacked_labels)
