from torch import optim
import torch
import torch.nn.functional as F

from ser.model import Net
from ser.transforms import transforms, normalize, flip

import json

from ser.transforms import normalize


def train(run_path, params, train_dataloader, val_dataloader, device, plotter):
    # setup model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # train
    val_acc = 0
    best_epoch = 0
    for epoch in range(params.epochs):
        train_loss, _ = _train_batch(model, train_dataloader, optimizer, epoch, device)
        val_loss, val_accuracy = _val_batch(model, val_dataloader, device, epoch)

        # Plot loss after all mini-batches have finished
        plotter.plot('loss', 'train', 'Class Loss', epoch, train_loss.item())
        plotter.plot('loss', 'val', 'Class Loss', epoch, val_loss)
        plotter.plot('acc', 'val', 'Class Accuracy', epoch, val_accuracy)

        if val_accuracy > val_acc:
            val_acc = val_accuracy
            best_epoch = epoch
            print("New best model at epoch {epoch}")

            # save model and save model params
            torch.save(model, run_path / "model.pt")

    with open(run_path / "best_results.json", "w") as file:
        json.dump({"epoch":best_epoch, "accuracy":val_acc}, file, indent = 4) 

    


def _train_batch(model, dataloader, optimizer, epoch, device):
    train_loss = 0
    correct = 0
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(images)
        train_loss = F.nll_loss(output, labels)
        train_loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        train_accuracy = correct / len(dataloader.dataset)

        print(
            f"Train Epoch: {epoch} | Batch: {i}/{len(dataloader)} "
            f"| Loss: {train_loss.item():.4f} "
            f"| Accuracy: {train_accuracy}"
        )
    
    return train_loss, train_accuracy

@torch.no_grad()
def _val_batch(model, dataloader, device, epoch):
    val_loss = 0
    correct = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        model.eval()
        output = model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(dataloader.dataset)
    val_accuracy = correct / len(dataloader.dataset)
    print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_accuracy}")

    return val_loss, val_accuracy
