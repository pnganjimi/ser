from ser import transform, data
from ser.model import Net
from ser.train import training, validation

import torch
from torch import optim

from pathlib import Path
from dataclasses import dataclass, asdict
import typer
import json
from datetime import datetime
import os

main = typer.Typer()
PROJECT_ROOT = Path(__file__).parent.parent

@dataclass
class Parameters:
    """Class for keeping track of an item in inventory."""
    name: str
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str



@main.command()
def train(
    name: str = typer.Option(
        ..., "-n", "--name", help="Name of experiment to save under."
    ),
    epochs: int = typer.Option(
        1, "-e", "--epochs", help="The number of epochs"
    ),
    batch_size: int = typer.Option(
        4000, "-b", "--batch_size", help="The batch size for each training run."
    ),
    learning_rate: float = typer.Option(
        0.01, "-lr", "--learning_rate", help="The learning rate of the optimiser."
    )
    ,
):

    # Save the parameters!
    params = Parameters(name, epochs, batch_size, learning_rate, 'Adam')


    print(f"Running experiment {params.name}")

    # Set training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directory for experiment outputs
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_time = ("_").join(current_time.split())

    exp_dir = (PROJECT_ROOT / "runs" / params.name / exp_time)
    os.makedirs(exp_dir)

    with open(exp_dir / "params.json", "w") as outfile:
        json.dump(asdict(params), outfile, indent = 4) 

    # Load model
    model = Net().to(device)

    # Setup params
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # Transform data
    ts = transform.tranformation()

    # Load the data
    training_dataloader, validation_dataloader = data.dataloader(params.batch_size, ts)

    val_acc = 0
    # Set up training loop
    for epoch in range(params.epochs):
        # Train the model
        print(f'Starting training for epoch {epoch}')
        training(epoch, device, model, optimizer, training_dataloader)

        # Validate the model
        print(f'Starting validation for epoch {epoch}')
        val_accuracy = validation(epoch, device, model, validation_dataloader)

        if val_accuracy > val_acc:
            val_acc = val_accuracy
            best_epoch = epoch
            print("New best model at epoch {epoch}")

    print("Training run complete")

    # Save trained model 
    torch.save(model, exp_dir / "model.pt")

    with open(exp_dir / "best_results.json", "w") as file:
        json.dump({"epoch":best_epoch, "accuracy":val_acc}, file, indent = 4) 



@main.command()
def infer():
    print("This is where the inference code will go")
