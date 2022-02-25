import os
import awkward as ak
import numpy as np
import uproot
from typing import List, Tuple
from source.feature_handler import FeatureHandler
import gc
from copy import deepcopy

class DataLoader:

    def __init__(self, files: List[str], features: FeatureHandler, batch_size: int=1) -> None:
        
        self.files = files
        self.features = features

        self.batch_size = batch_size
        self.is_jet = False
        if "JZ" in os.path.basename(files[0][0]):
            self.is_jet = True
    
        self.lazy_array = uproot.lazy(self.files, filter_name=self.features.as_list(), step_size="50 MB")
        # self.lazy_array = np.ones(int(1e6))

    def __getitem__(self, idx) -> Tuple:

        # batch = deepcopy(self.lazy_array[idx * self.batch_size: (idx  + 1) * self.batch_size])
        # batch = self.lazy_array[idx * self.batch_size: (idx  + 1) * self.batch_size]#.copy()

        # return batch, batch, batch

        batch = self.lazy_array[idx * self.batch_size: (idx  + 1) * self.batch_size]

        tracks = ak.unzip(batch[self.features["TauTracks"]])
        tracks = np.stack([ak.to_numpy(ak.pad_none(arr, 3, clip=True)) for arr in tracks], axis=1).filled(0)  

        neutral_pfo = ak.unzip(batch[self.features["NeutralPFO"]])
        neutral_pfo = np.stack([ak.to_numpy(ak.pad_none(arr, 6, clip=True)) for arr in neutral_pfo], axis=1).filled(0)    

        shot_pfo = ak.unzip(batch[self.features["ShotPFO"]])
        shot_pfo = np.stack([ak.to_numpy(ak.pad_none(arr, 8, clip=True)) for arr in shot_pfo], axis=1).filled(0)      
        
        conv_tracks = ak.unzip(batch[self.features["ConvTrack"]])
        conv_tracks = np.stack([ak.to_numpy(ak.pad_none(arr, 4, clip=True)) for arr in conv_tracks], axis=1).filled(0)         

        jets = ak.unzip(batch[self.features["TauJets"]])
        jets = np.stack([ak.to_numpy(arr) for arr in jets], axis=1)     

        if not self.is_jet:
            decay_mode = ak.to_numpy(batch["TauJets.truthDecayMode"])
            labels = np.zeros((len(decay_mode), 6))  
            for i, dm in enumerate(decay_mode):
                labels[i][dm] += 1
        else:
            labels = np.zeros((len(jets), 6))
            labels[:, 0] = 1

        weights = ak.to_numpy(batch["TauJets.mcEventWeight"])

        return ((tracks, neutral_pfo, shot_pfo, conv_tracks, jets), labels, weights)

    def __len__(self) -> int:
        return len(self.lazy_array) // self.batch_size
