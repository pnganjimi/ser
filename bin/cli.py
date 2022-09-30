from ser import transform, data
from ser.model import Net
from ser.train import training, validation
from pathlib import Path
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import typer

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        2, "-e", "--epochs", help="The number of epochs"
    ),
    batch_size: int = typer.Option(
        1000, "-b", "--batch_size", help="The batch size for each training run."
    ),
    learning_rate: float = typer.Option(
        0.01, "-lr", "--learning_rate", help="The learning rate of the optimiser."
    )
    ,
):

    print(f"Running experiment {name}")

    # Set training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save the parameters!

    # Load model
    model = Net().to(device)

    # Setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Transform data
    ts = transform.tranformation()

    # Load the data
    training_dataloader, validation_dataloader = data.dataloader(batch_size, DATA_DIR, ts)

    # Set up training loop
    for epoch in range(epochs):
        # Train the model
        print(f'Starting training for epoch {epoch}')
        training(epoch, device, model, optimizer, training_dataloader)

        # Validate the model
        print(f'Starting validation for epoch {epoch}')
        validation(epoch, device, model, validation_dataloader)

    print("Training run complete")



@main.command()
def infer():
    print("This is where the inference code will go")
