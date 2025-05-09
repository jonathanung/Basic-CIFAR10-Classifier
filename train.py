import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from CNN import CNN

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    augmented_transform = transforms.Compose([
        # introduce random cropping to augment the training data
        transforms.RandomCrop(32, padding=4), 
        # im2tensor
        transforms.ToTensor(),
        # normalize images
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=augmented_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True)

    # Initialize model
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # training loop
    for epoch in range(5):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[Epoch {epoch+1}, Batch {i+1}] loss: {running_loss/100:.3f}')
                running_loss = 0.0

    # save model
    torch.save(model.state_dict(), 'cnn.pth')
    print('trained model saved as cnn.pth')

if __name__ == "__main__":
    main()

