import torch
import torch.nn as nn
import torch.optim as optim
import time
from dataset import get_dataloaders
from model import get_model

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 1 # As running locally on CPU, reducing this value to 1 from 3
    save_path = "resnet18_cifar10.pth"

    # Get Data and Model
    trainloader, testloader = get_dataloaders(batch_size=batch_size)
    model = get_model(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    print("Starting Training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for i, (inputs, labels) in enumerate(trainloader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:    
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        print(f"Epoch {epoch + 1} time: {time.time() - start_time:.2f}s")

    # Save the trained weights
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    train()