import gc
from math import ceil
import numpy as np
import uproot
import awkward as ak
from typing import List, Tuple, Dict
from source.feature_handler import FeatureHandler  # Simple class to help manage input variables


class DataGenerator:

    # A complete version of this class for training NNs will inherit from tf.keras.utils.Sequence

    def __init__(self, files_dict: Dict[str, List[str]], feature_handler: FeatureHandler, batch_size: int) -> None:

        lazy_arrays = {}
        self.batch_sizes = {}
        self.iters = {}
        self.batch_size = batch_size
        self.features = feature_handler
        self.files_dict = files_dict

        # Make some lazy arrays to help work out sizes of each dataset
        for file_type, file_list in files_dict.items():
            lazy_arrays[file_type] = uproot.lazy(file_list, step_size="50 MB")

        # Work out the step size for each iterator - each full batch should have the same proportion of 
        # each file type
        self.nevents = sum([len(arr) for arr in lazy_arrays.values()])
        print(self.nevents)

        for file_type, array in lazy_arrays.items():
            self.batch_sizes[file_type] = ceil(len(array) / self.nevents * batch_size)
        
        # Create iterators
        self.create_itrs()
    

    def create_itrs(self) -> None:

        # Create a dictionary of uproot.iterate generators
        print("Create itrs called")
        for file_type, file_list in self.files_dict.items():
            self.iters[file_type] = None
            gc.collect()
            self.iters[file_type] = uproot.iterate(file_list, step_size=self.batch_sizes[file_type], filter_name=self.features.as_list(),
                                                    file_handler=uproot.MultithreadedFileSource)


    def process_batch(self, batch: ak.Array, is_fake: bool=False) -> Tuple:
        
        # Process one batch of data yielded by uproot.iterate into input features, labels and weights
        # Network has 5 inputs tracks, neutral PFOs, shot PFOs, conversion tracks and high-level jet info

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

        # Compute labels: [1, 0 ,0 ,0 ,0, 0] = fake 
        #                 [0, 1 ,0 ,0 ,0, 0] = 1p0n
        #                 [0, 0 ,1 ,0 ,0, 0] = 1p1n
        #                 [0, 0 ,0 ,1 ,0, 0] = 1pXn
        #                 [0, 0 ,0 ,0 ,1, 0] = 3p0n
        #                 [0, 0 ,0 ,0 ,0, 1] = 1pXn                  
        if not is_fake:
            decay_mode = ak.to_numpy(batch["TauJets.truthDecayMode"])
            labels = np.zeros((len(decay_mode), 6))
            for i, dm in enumerate(decay_mode):
                # Note: truth decay mode runs from 0 -> 4, but we are using index 0 for fakes
                labels[i][int(dm + 1)] += 1  
        else:
            labels = np.zeros((len(jets), 6))
            labels[:, 0] = 1

        # Get weights
        weights = ak.to_numpy(batch["TauJets.mcEventWeight"])

        return ((tracks, neutral_pfo, shot_pfo, conv_tracks, jets), labels, weights)
    
    def __getitem__(self, idx: int) -> Tuple:

        # Use __getitem__ to fetch batch since this is what tf.keras.utils.Sequences uses
        # Looks like a weird way to do things but this is how tensorflow likes it

        gc.collect()

        # Get sub-batches from each iterator
        # batch = []
        for file_type, itr in self.iters.items():
            is_fake = True
            # Check if itr is tau/fake so that labels are computed correctly
            if "Gammatautau" in file_type:
                is_fake = False 
            try:
                array = next(itr)
            except StopIteration:
                # If we reach the end of an iterator before the end of an epoch, recreate that iterator and try again
                self.iters[file_type] = None
                gc.collect()
                self.iters[file_type] = uproot.iterate(self.files_dict[file_type], step_size=self.batch_sizes[file_type], filter_name=self.features.as_list(),
                                                    file_handler=uproot.MultithreadedFileSource)
                print(f"Remade {file_type} iterator")
                array = next(self.iters[file_type])
            # batch.append(self.process_batch(array, is_fake))


        # Concatenate all sub-batches into a single batch
        # x_batch = []
        # for i in range(0, len(batch[0])):
        #     x_batch.append(np.concatenate([result[0][i] for result in batch]))

        # y_batch = np.concatenate([result[1] for result in batch])
        # weight_batch = np.concatenate([result[2] for result in batch])

        # return x_batch, y_batch, weight_batch
        
    def __len__(self) -> None:
        # Number of iterations to do one complete pass of the dataset
        return ceil(self.nevents / self.batch_size)
    
    def reset(self) -> None:
        print("Reset called")
        self.create_itrs()