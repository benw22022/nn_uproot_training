
from typing import List
from dataclasses import dataclass

@dataclass
class Feature:
    # Helper class to handle input features
    name: str
    mean: int = None
    std: int = None

class FeatureHandler:
    """
    Class to handle training features and other input variables
    """
    def __init__(self) -> None:
        self.branches = {}

    def add_branch(self, branch_name: str, features: List[Feature]) -> None:
        # Add another key, value pair to branches dictionary
        self.branches[branch_name] = features

    def __getitem__(self, key: str):
        # Returns a list of strings containing the names of features in this branch 
        return [f.name for f in self.branches[key]]
    
    def as_list(self) -> List[str]:
        # Return a list of all input variables in all branches 
        # - useful for filter_name option in uproot
        names = []
        for value in self.branches.values():
            if isinstance(value, list):
                for f in value:
                    names.append(f.name)
            else:
                names.append(value.name)
        return list(set(names))

    def get_stats(self, branch_name: str, feature_name: str):
        # Get mean and standard devation for a feature belonging to a specific branch
        for f in self.branches[branch_name]:
            if f.name == feature_name:
                return f.mean, f.std
