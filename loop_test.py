"""
Loop over generator
________________________________________________
Script to just loop over batch generator for
debugging 
"""
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import tqdm
# from source.datagenerator import DataGenerator
from source.complex_datagenerator import DataGenerator
# from source.dataloader import DataLoader
from config.config import config_dict, models_dict
from config.features import make_feature_handler
import subprocess


def grab_files(directory, glob_exprs):
    files = []
    for expr in glob_exprs: 
        files.append([glob.glob(os.path.join(directory, expr, "*.root"))])
        
    return files

def train():

    files = grab_files("../NTuples/", ["*Gammatautau*", "*JZ1*", "*JZ2*", "*JZ3*", "*JZ4*", "*JZ5*", "*JZ6*", "*JZ7*", "*JZ8*"])
    
    files_dict = {"Gammatautau": files[0], 
                  "JZ1": files[1],
                  "JZ2": files[2],
                  "JZ3": files[3],
                  "JZ4": files[4],
                  "JZ5": files[5],
                  "JZ6": files[6],
                  "JZ7": files[7],
                  "JZ8": files[8],
                  }

    training_batch_generator = DataGenerator(files_dict, make_feature_handler(), batch_size=100000)

    while True:
        for i in tqdm.tqdm(range(0, len(training_batch_generator))):
            x = training_batch_generator[i]


if __name__ == "__main__":
    
    loop_test()