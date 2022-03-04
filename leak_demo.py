"""
Loop over generator
__________________________________________________________
Script to just loop over batch generator for debugging 
"""
import glob
import tqdm
from source.datageneratorMK2 import DataGenerator     # Uses uproot.iterate
# from source.datagenerator import DataGenerator      # Uses uproot.lazy
from config.features import make_feature_handler


def simulate_training_loop():

    # List of files - one of each type
    files = glob.glob("test_files/*.root")
    
    # Note: DataGenerator expects a dict with str: List[str] structure
    # file type keys are generic - correct names not needed for this test
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

    # Create Generator - use large batch size to speed up iterations (would typically use much smaller one for training)
    training_batch_generator = DataGenerator(files_dict, make_feature_handler(), batch_size=100000)

    # Simulate the training loop
    for _ in range(200):
        for i in tqdm.tqdm(range(0, len(training_batch_generator))):
            x = training_batch_generator[i]
        training_batch_generator.reset()

if __name__ == "__main__":
    
    simulate_training_loop()