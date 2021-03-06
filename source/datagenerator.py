import numpy as np
from typing import List, Dict
from source.feature_handler import FeatureHandler
from source.dataloader import DataLoader
from math import ceil

# class DataGenerator(tf.keras.utils.Sequence):
class DataGenerator:

    def __init__(self, files_dict: Dict[str, List[str]], feature_handler: FeatureHandler, batch_size: int) -> None:
        
        self.dataloaders = []
        self.batch_size = batch_size

        for file_list in files_dict.values():
            self.dataloaders.append(DataLoader(file_list, feature_handler))

        dataloader_lengths = [len(dl) for dl in self.dataloaders]
        self.nevents = sum(dataloader_lengths)

        for dl in self.dataloaders:
            dl.batch_size = len(dl) // self.nevents * batch_size

    def __getitem__(self, idx):
        batch = [dl[idx] for dl in self.dataloaders]
        x_batch = []
        for i in range(0, len(batch[0])):
            x_batch.append(np.concatenate([result[0][i] for result in batch]))

        y_batch = np.concatenate([result[1] for result in batch])
        weight_batch = np.concatenate([result[2] for result in batch])

        return x_batch, y_batch, weight_batch
    

    def __len__(self):
        return ceil(self.nevents / self.batch_size)