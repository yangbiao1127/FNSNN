import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def function(x, a1, t1, b1):
    return a1 * np.exp(-x / t1) + b1

class FNEncoder:
    def __init__(self, a1, t1, b1):
        self.a1 = a1
        self.t1 = t1
        self.b1 = b1

    def encode(self, image: torch.Tensor):
        image = image.view(image.size(0), -1)
        pulse_frequencies = function(image.numpy(), a1=self.a1, t1=self.t1, b1=self.b1)
        pulse_frequencies = torch.tensor(pulse_frequencies).float()
        pulse_frequencies = pulse_frequencies.view(image.size(0), 1, 28, 28)
        return pulse_frequencies


class FNNeuron(nn.Module):
    def __init__(self, a= , b= , c= , dt= , I_ext= ):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.dt = dt
        self.I_ext = I_ext
        self.v = torch.zeros(1, 128)
        self.w = torch.zeros(1, 128)

    def forward(self, x: torch.Tensor):

        dv = self.v - (self.v ** 3) / 3 - self.w + self.I_ext + x
        self.v += dv * self.dt


        self.w += (self.v - self.a - self.b * self.w) / self.c * self.dt


        spike = self.v >= 1.0  # Spike threshold
        self.v.masked_fill_(spike, 0.0)  # Reset potential after spike
        return spike.float()


class FHNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FNEncoder(a1=-1.405e23, t1=0.02012, b1=5.9595e7)
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fhn_neuron1 = FNNeuron()
        self.fc2 = nn.Linear(128, 20)
        self.fhn_neuron2 = FNNeuron()

    def forward(self, x: torch.Tensor):
        x = self.encoder.encode(x).view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fhn_neuron1(x)
        x = self.fc2(x)
        x = self.fhn_neuron2(x)
        return x


def load_data_from_folder(batch_size=64, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def train_snn(net, train_loader, optimizer, criterion, num_epochs=100):
    train_accuracies = []
    for epoch in range(num_epochs):
        net.train()
        total_loss = 0
        correct = 0
        total = 0
        for img, label in train_loader:
            img = img.view(img.size(0), -1).cuda()
            label = label.cuda()

            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


            _, predicted = outputs.max(1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

        accuracy = correct / total * 100
        train_accuracies.append(accuracy)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")
    return train_accuracies


def atest_snn(net, test_loader):
    net.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for img, label in test_loader:
            img = img.view(img.size(0), -1).cuda()
            label = label.cuda()
            outputs = net(img)
            _, predicted = outputs.max(1)
            correct += (predicted == label).sum().item()
            total += label.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    accuracy = correct / total * 100
    print(f"Test Accuracy: {accuracy:.2f}%")


    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_loader.dataset.classes,
                yticklabels=test_loader.dataset.classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


if __name__ == "__main__":
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-3

    train_loader, test_loader = load_data_from_folder(batch_size=batch_size)

    net = FHNNetwork().cuda()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_accuracies = train_snn(net, train_loader, optimizer, criterion, num_epochs=num_epochs)
    atest_snn(net, test_loader)
