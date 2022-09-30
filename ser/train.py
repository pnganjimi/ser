import torch
import torch.nn.functional as F



# Training run over a single epoch
def training(epoch, device, model, optimizer, training_dataloader):
    '''Train the model for a single epoch'''

    for i, (images, labels) in enumerate(training_dataloader):
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        print(
            f"Train Epoch: {epoch} | Batch: {i}/{len(training_dataloader)} "
            f"| Loss: {loss.item():.4f}"
            )

    

            
# Validate run over a single epoch
def validation(epoch, device, model, validation_dataloader):
    '''Validate a trained model'''
    
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in validation_dataloader:
            images, labels = images.to(device), labels.to(device)
            model.eval()
        output = model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        val_loss /= len(validation_dataloader.dataset)
        val_acc = correct / len(validation_dataloader.dataset)

        print(
            f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {val_acc}"
            )