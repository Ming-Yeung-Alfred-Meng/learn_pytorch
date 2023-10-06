import torch
import torchvision.utils
from torch.utils.data import Dataset
from torch import nn
import torchvision
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(nn.Linear(28 * 28, 512),
                                               nn.ReLU(),
                                               nn.Linear(512, 512),
                                               nn.ReLU(),
                                               nn.Linear(512, 10))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model: nn.Module, loss_fn, optimizer, device: str):
    model.train()

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x)
            print(f"loss: {loss:>7f} [{current:>5d}/{len(dataloader.dataset):>5d}]")


def train_with_validation(dataloader, model: nn.Module, loss_fn, optimizer, device: str) -> None:
    model.train()
    running_loss = 0.0

    for batch, (x, y) in enumerate(dataloader):
        x, y = x.to(device), y.to(device)

        loss = loss_fn(model(x), y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        if batch % 100 == 0:
            running_validation_loss = 0.0

            model.eval()

            for j, (validation_images, validation_labels) in enumerate(dataloader):
                validation_images = validation_images.to(device)
                validation_labels = validation_labels.to(device)

                running_validation_loss += loss_fn(model(validation_images), validation_labels)

            model.train()

            writer.add_scalars("Training v.s. Validation Loss",
                               {"Training"})



def test(dataloader, model: nn.Module, loss_fn, device: str):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= len(dataloader)
    correct /= len(dataloader.dataset)

    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def peek_dataset(dataset, classes, rows: int, columns: int):
    figure = plt.figure(figsize=(8, 8))

    for i in range(1, rows * columns + 1):
        sample_id = torch.randint(len(dataset), size=(1,), dtype=torch.int64).item()

        image, label = dataset[sample_id]

        figure.add_subplot(rows, columns, i)
        plt.title(classes[label])
        plt.axis("off")
        plt.imshow(image.squeeze(), cmap="gray")

    plt.show()


def image_grid_from_dataset(dataset: Dataset, rows: int, columns: int) -> torch.tensor:
    image_list = []

    for i in range(1, rows * columns + 1):
        sample_id = torch.randint(len(dataset), size=(1,), dtype=torch.int64).item()
        image, label = dataset[sample_id]
        image_list.append(image)

    return torchvision.utils.make_grid(torch.stack(image_list, dim=0), nrow=rows)