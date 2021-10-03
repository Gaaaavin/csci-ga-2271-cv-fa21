# -*- coding: utf-8 -*

# Commented out IPython magic to ensure Python compatibility.
# Load the Drive helper and mount
# %cd  /content/drive/MyDrive/Colab Notebooks/Computer Vision Assignment/Assignment 1/dataset

"""# Dataloader"""

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import pickle
from torchsummary import summary

batch_size = 32
momentum = 0.9
lr = 1e-3
epochs = 50
log_interval = 500
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyDataset(Dataset):

    def __init__(self, X_path="dataset/X.pt", y_path="dataset/y.pt"):

        self.X = torch.load(X_path).squeeze(1)
        self.y = torch.load(y_path).squeeze(1)
    
    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = MyDataset(X_path="dataset/train/X.pt", y_path="dataset/train/y.pt")
val_dataset = MyDataset(X_path="dataset/validation/X.pt", y_path="dataset/validation/y.pt")

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

transform = transforms.Grayscale()

"""# Model"""

import torch
import torch.nn as nn
import torch.nn.functional as F

nclasses = 43 # GTSRB has 43 classes

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc1 = nn.Sequential(nn.Flatten(), nn.Linear(4096, 512), nn.BatchNorm1d(512), nn.LeakyReLU(), nn.Dropout())
        self.fc2 = nn.Linear(512, 43)

    # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 128 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x    

    def forward(self, x):
        x = self.stn(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

"""# Training"""

model = LeNet()
model = model.to(device)
summary(model, (3, 32, 32))
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

train_loss = np.zeros(epochs)
train_accuracy = np.zeros(epochs)
valid_loss = np.zeros(epochs)
valid_accuracy = np.zeros(epochs)

print("Let's use {} gpu(s)!".format(torch.cuda.device_count()))
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
valid_criterion = nn.CrossEntropyLoss(reduction='sum')

def train(epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        # data = transform(data)
        target = target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    scheduler.step()
    
    return loss.item(), 100. * correct / len(train_loader.dataset)

def validation():
    model.eval()
    validation_loss = 0
    correct = 0
    for data, target in val_loader:
        data = data.to(device)
        # data = transform(data)
        target = target.to(device)
        output = model(data)
        validation_loss += valid_criterion(output, target).item() # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    validation_loss /= len(val_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        validation_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
    return validation_loss, 100. * correct / len(val_loader.dataset)


for epoch in range(1, epochs + 1):
    train_loss[epoch - 1], train_accuracy[epoch - 1] = train(epoch)
    valid_loss[epoch - 1], valid_accuracy[epoch - 1] = validation()
    if epoch > epochs - 8:
        outfile = 'lestn_{}.csv'.format(epoch)

        output_file = open(outfile, "w")
        dataframe_dict = {"Filename" : [], "ClassId": []}
        test_data = torch.load('dataset/testing/test.pt')
        file_ids = pickle.load(open('dataset/testing/file_ids.pkl', 'rb'))
        for i, data in enumerate(test_data):
            data = data.unsqueeze(0)
            data = data.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1].item()
            file_id = file_ids[i][0:5]
            dataframe_dict['Filename'].append(file_id)
            dataframe_dict['ClassId'].append(pred)

        df = pd.DataFrame(data=dataframe_dict)
        df.to_csv(outfile, index=False)
        print("Written to csv file {}".format(outfile))

x = np.arange(1, epochs + 1, 1)
plt.plot(x[10:], train_loss[10:], label='train');
plt.plot(x[10:], valid_loss[10:], label='validation');
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend();
plt.title('Train/Validation Loss')
plt.savefig('loss.png')
plt.clf()
plt.plot(x[10:], train_accuracy[10:], label='train');
plt.plot(x[10:], valid_accuracy[10:], label='validation');
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend();
plt.title('Train/Validation Accuracy')
plt.savefig('accu.png')
