from datetime import datetime
from pathlib import Path
from typing import List
import tempfile

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import git

from ser.train import train as run_train
from ser.infer import infer as run_infer
from ser.constants import RESULTS_DIR, DATA_DIR
from ser.params import Params, save_params, load_params
from ser.transforms import transforms, normalize, flip
from ser.utils import VisdomLinePlotter


def train_dataloader(batch_size, transforms):
    data = datasets.MNIST(
        root=DATA_DIR, download=True, train=True, transform=transforms
    )
    subset = Subset(data, (0,10))
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=1)


def val_dataloader(batch_size, transforms):
    data = datasets.MNIST(
        root=DATA_DIR, download=True, train=False, transform=transforms
    )
    subset = Subset(data, (0,10))
    return DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=1)


def eval_dataloader(batch_size, transforms):
    data = datasets.MNIST(
        root=DATA_DIR, download=True, train=False, transform=transforms
    )
    subset = Subset(data, (0,10))
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=1)

# setup params
name = 'test_experiment'
epochs = 1
batch_size = 10
learning_rate = 0.01
ts = [normalize, flip]


def test_e2e():
    with tempfile.TemporaryDirectory() as tmpdir:
        """Run the training algorithm."""
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha

        # setup run
        fmt = "%Y-%m-%dT%H-%M"
        timestamp = datetime.strftime(datetime.utcnow(), fmt)
        run_path = Path(tmpdir) / name / timestamp
        run_path.mkdir(parents=True, exist_ok=True)

        # wraps the passed in parameters
        params = Params(name, epochs, batch_size, learning_rate, sha)

        # setup device to run on
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Save parameters for the run
        save_params(run_path, params)

        global plotter
        plotter = VisdomLinePlotter(env_name='Tutorial Plots')

        # Train!
        run_train(
            run_path,
            params,
            train_dataloader(params.batch_size, transforms(normalize, flip)),
            val_dataloader(params.batch_size, transforms(normalize)),
            device,
            plotter,
            )

        Path.exists(run_path / "model.pt")
        Path.exists(run_path / "params.json")
        Path.exists(run_path / "best_results.json")


        """Run the inference code"""
        label = 6
        infer_params = load_params(run_path)
        model = torch.load(run_path / "model.pt")
        image = _select_test_image(label, ts)
        run_infer(infer_params, model, image, label)


    def _select_test_image(label, ts):
        dataloader = eval_dataloader(1, transforms(*ts))
        images, _ = next(iter(dataloader))
        return images

