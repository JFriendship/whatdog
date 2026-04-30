import torch
import torch.nn as nn
import torchvision.models as models # For loading resnet18
from datetime import datetime


def get_device():
    """
    Uses the best available device for pytorch operations.
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() 
        else "mps" if torch.backends.mps.is_available() 
        else "cpu"
    )

    return device

def load_and_finetune_resnet18():
    """
    Loads resnet18 and fine-tunes the final classification layer to predict 120 classes.
    Returns:
        ResNet: The fine-tuned resnet18 model
    """

    # Load the pretrained Resnet18 model
    res18_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze the model parameters
    for param in res18_ft.parameters():
        param.requires_grad = False

    # Unfreeze the last layer stage of resnet18
    for param in res18_ft.layer4.parameters():
        param.requires_grad = True

    # Update the final classification layer (updatable parameters)
    num_classes = 120
    res18_ft.fc = nn.Linear(res18_ft.fc.in_features, num_classes)

    return res18_ft

def evaluate_accuracy(model, val_loader, device=get_device()):
    """
    Evaluates the multi-class accuracy of a model using a validation dataloader.
    Args:
        model (ResNet): The model to evaluate
        val_loader (DataLoader): The DataLoader used for validation.
        device (device): The device that will perform the model operations.
    """

    model.to(device)
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, classification = outputs.max(1)

            total += labels.size(0)
            correct += classification.eq(labels).sum().item()

    print(f"{correct=} \n{total=}")
    return 100. * correct / total

def training_loop(model, train_loader, loss_fn, optimizer, num_epochs=5, device=get_device()):
    """
    Trains a model using the provided arguments and hyperparameters.
    Arguments:
    model (nn.Module): A pytorch model
    train_loader (Dataloader): A pytorch DataLoader that is loaded with the training data 
    loss_fn: A loss function used to calculate the loss
    optimizer: An optimizer for updating the weights of the model
    """
    model.to(device)
    model.train()
    running_loss = 0
    num_batches = 0

    for epoch in range(1, num_epochs+1):
        running_loss = 0
        num_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss
            num_batches += 1

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'date': datetime.now().isoformat()
        }
        
        print(f"EPOCH {epoch} loss: {running_loss / num_batches}")
        if epoch % 5 == 0:
            torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pt')
            print("=== Model Saved ===")

def load_model_from_pt(model, optimizer, checkpoint_file_name: str):
    checkpoint = torch.load(checkpoint_file_name)
    model_state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(model_state_dict)
    optimizer_state_dict = checkpoint.get('optimizer_state_dict', checkpoint)
    optimizer.load_state_dict(optimizer_state_dict)

def save_model(model, optimizer, epochs_trained: int, learning_rate: float, train_batch_size: int, val_batch_size: int, accuracy: float, note: str, save_file_path: str):
    save_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs_trained': epochs_trained,
        'learning_rate': learning_rate,
        'train_batch_size': train_batch_size,
        'val_batch_size': val_batch_size,
        'accuracy': accuracy,
        'note': note
    }
    torch.save(save_data, save_file_path)