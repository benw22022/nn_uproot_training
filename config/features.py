from source import feature_handler as fh
import pandas as pd
import yaml

def make_feature_handler():
    # Function to read yaml and csv files to initialize FeatureHandler

    stats_df = pd.read_csv("config/stats_df.csv", index_col=0)

    feature_handler = fh.FeatureHandler()

    with open("config/features.yaml", 'r') as stream:

        for key, value in yaml.load(stream, Loader=yaml.FullLoader).items():
            
            if isinstance(value, list):
                tmp_features = []
                for item in value:
                    mean = stats_df.iloc[list(stats_df.index).index(item)]["Mean"]
                    std = stats_df.iloc[list(stats_df.index).index(item)]["StdDev"]
                    tmp_features.append(fh.Feature(item, mean, std))

                feature_handler.add_branch(key, tmp_features)
            # else:
            #     feature_handler.add_branch(key, fh.Feature(value))
    
    return feature_handler