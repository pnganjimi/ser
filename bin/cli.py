from ser import transform, data
from ser.model import Net
from ser.train import training, validation

import torch
from torch import optim

from pathlib import Path
import typer
import json
from datetime import datetime
import os

main = typer.Typer()

PROJECT_ROOT = Path(__file__).parent.parent

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

    # Create directory for experiment outputs
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exp_time = current_time.split()[0] + "_" + current_time.split()[1]

    exp_dir = (PROJECT_ROOT / "runs" / name / exp_time)
    os.makedirs(exp_dir)

    # Save the parameters!
    params = {'name': name,
              'epochs': epochs,
              'batch_size': batch_size,
              'learning_rate': learning_rate,
              'optimizer': 'Adam'

            }

    with open(exp_dir / "params.json", "w") as outfile:
        json.dump(params, outfile, indent = 4) 

    # Load model
    model = Net().to(device)

    # Setup params
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Transform data
    ts = transform.tranformation()

    # Load the data
    training_dataloader, validation_dataloader = data.dataloader(batch_size, ts)

    # Set up training loop
    for epoch in range(epochs):
        # Train the model
        print(f'Starting training for epoch {epoch}')
        training(epoch, device, model, optimizer, training_dataloader)

        # Validate the model
        print(f'Starting validation for epoch {epoch}')
        validation(epoch, device, model, validation_dataloader)

    print("Training run complete")


    # Save trained model 
    torch.save(model, exp_dir / "model.pt")

    

@main.command()
def infer():
    print("This is where the inference code will go")
