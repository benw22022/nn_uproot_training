
from typing import List
from dataclasses import dataclass

@dataclass
class Feature:
    name: str
    mean: int = None
    std: int = None

class FeatureHandler:

    def __init__(self) -> None:
        self.branches = {}

    def add_branch(self, branch_name: str, features: List[Feature]) -> None:
        self.branches[branch_name] = features

    def __getitem__(self, key: str):
        return [f.name for f in self.branches[key]]
    
    def as_list(self) -> List[str]:
        names = []
        for value in self.branches.values():
            if isinstance(value, list):
                for f in value:
                    names.append(f.name)
            else:
                names.append(value.name)
        return names

    def get_stats(self, branch_name: str, feature_name: str):
        for f in self.branches[branch_name]:
            if f.name == feature_name:
                return f.mean, f.std
