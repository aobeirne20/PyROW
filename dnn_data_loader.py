import os
import torch
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from imblearn.over_sampling import SMOTE

PyROW_year_data = {
    #2010: "PyROW_data_year2010.parquet",
    #2011: "PyROW_data_year2011.parquet"
    #2012: "PyROW_data_year2012.parquet",
    #2013: "PyROW_data_year2013.parquet",
    #2014: "PyROW_data_year2014.parquet",
    #2015: "PyROW_data_year2015.parquet",
    #2016: "PyROW_data_year2016.parquet",
    2017: "PyROW_data_year2017.parquet",
    2018: "PyROW_data_year2018.parquet",
    2019: "PyROW_data_year2019.parquet",
    2020: "PyROW_data_year2020.parquet",
    #2021: "PyROW_data_year2021.parquet"
}

PyROW_test_data = {
    2021: "PyROW_data_year2021.parquet"
}

generic_columns = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'AGENCY', 'GROUNDCOVER', 'PRECIPITATION', 'TEMPERATURE', 'DROUGHT', 'FIRE']


class dnnDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, test=False):
        if test:
            n = 100000
            data_to_use = PyROW_year_data
        else:
            n = 10000000
            data_to_use = PyROW_test_data
        loaded_data = {}
        for year, text in data_to_use.items():
            loaded_data[year] = pd.read_parquet(text).sample(n=n)
            loaded_data[year].columns = generic_columns

        lst = [[0, 0, 0, 'None', 'None', 0, 0, 'No Drought', 0],
               [0, 0, 0, 'None', 'None', 0, 0, 'D0', 0],
               [0, 0, 0, 'None', 'None', 0, 0, 'D1', 0],
               [0, 0, 0, 'None', 'None', 0, 0, 'D2', 0],
               [0, 0, 0, 'None', 'None', 0, 0, 'D3', 0],
               [0, 0, 0, 'None', 'None', 0, 0, 'D4', 0],
               [0, 0, 0, 'None', "Unclassified", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Open Water", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Perennial Snow/Ice", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Developed, Open Space", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Developed, Low Intensity", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Developed, Medium Intensity", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Developed, High Intensity", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Barren Land", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Deciduous Forest", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Evergreen Forest", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Mixed Forest", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Herbaceous", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Hay/Pasture", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Cultivated Crops", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Woody Wetlands", 0, 0, 'D3', 0],
               [0, 0, 0, 'None', "Emergent Herbaceous Wetlands", 0, 0, 'D3', 0],
               [0, 0, 0, "No Agency", 'None', 0, 0, 'D3', 0],
               [0, 0, 0, "Fish and Wildlife Service", 'None', 0, 0, 'D3', 0],
               [0, 0, 0, "Bureau of Land Management", 'None', 0, 0, 'D3', 0],
               [0, 0, 0, "National Park Service", 'None', 0, 0, 'D3', 0],
               [0, 0, 0, "Forest Service", 'None', 0, 0, 'D3', 0],
               [0, 0, 0, "Bureau of Reclamation", 'None', 0, 0, 'D3', 0]]

        loaded_data[9999] = pd.DataFrame(lst, columns=generic_columns)

        master_df = pd.concat([*loaded_data.values()], axis=0)
        master_df = pd.get_dummies(master_df)
        master_df = master_df.drop(master_df.tail(28).index)

        self.master_input = master_df.drop(['FIRE'], axis=1)
        self.master_labels = master_df['FIRE']
        master_df = None

        x_resampled, y_resampled = SMOTE().fit_resample(self.master_input, self.master_labels)
        self.master_input = pd.DataFrame(x_resampled, columns=self.master_input.columns)
        self.master_labels = pd.Series(y_resampled)

        columns_to_norm = ['LATITUDE', 'LONGITUDE', 'ELEVATION', 'PRECIPITATION', 'TEMPERATURE']
        self.master_input[columns_to_norm] = (self.master_input[columns_to_norm]-self.master_input[columns_to_norm].min())/(self.master_input[columns_to_norm].max()-self.master_input[columns_to_norm].min())

        self.master_input = torch.tensor(self.master_input.values).float().to("cuda:0")
        self.master_labels = torch.tensor(self.master_labels.values).float().to("cuda:0")

        print(self.master_input.shape)
        print(self.master_labels.shape)

    def __len__(self):
        return len(self.master_labels)

    def __getitem__(self, idx):
        return self.master_input[idx], self.master_labels[idx]

