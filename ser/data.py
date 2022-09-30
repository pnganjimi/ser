from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def dataloader(batch_size, ts):
    
    training_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=True, transform=ts),
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    validation_dataloader = DataLoader(
        datasets.MNIST(root=DATA_DIR, download=True, train=False, transform=ts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    return training_dataloader, validation_dataloader
