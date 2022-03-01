"""
Loop over generator
________________________________________________
Script to just loop over batch generator for
debugging 
"""
import os
import glob
import tqdm
from source.datageneratorMK2 import DataGenerator
from config.features import make_feature_handler


def loop_test():

    files = glob.glob("../temp_files/*.root")
    
    files_dict = {"Type1": [files[0]], 
                  "Type2": [files[1]],
                  "Type3": [files[2]],
                  "Type4": [files[3]],
                  "Type5": [files[4]],
                  "Type6": [files[5]],
                  "Type7": [files[6]],
                  "Type8": [files[7]],
                  "Type9": [files[8]],
                  }

    training_batch_generator = DataGenerator(files_dict, make_feature_handler(), batch_size=100000)

    while True:
        for i in tqdm.tqdm(range(0, len(training_batch_generator))):
            x = training_batch_generator[i]
        training_batch_generator.reset()

if __name__ == "__main__":
    
    loop_test()