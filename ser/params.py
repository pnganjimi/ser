from dataclasses import dataclass, asdict
from typing import List
import json


@dataclass
class Params:
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    commit: str
    augmentations: List[str]


PARAMS_FILE = "params.json"


def save_params(run_path, params):
    with open(run_path / PARAMS_FILE, "w") as f:
        json.dump(asdict(params), f, indent=2)


def load_params(run_path):
    with open(run_path / PARAMS_FILE, "r") as f:
        return Params(**json.load(f))
