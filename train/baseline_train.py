import torch.optim as optim
from torch.utils.data import DataLoader
import sys

sys.path.append('..')
from shared.models import *
from shared.datasets import *

num_epochs = 2
num_workers = 8
bs = 64
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BaselineNet(10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_dataset = MimicCxrJpg(root='../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/',
                            csv_path='./splits.csv', mode='base_train', resize=224)
test_dataset = MimicCxrJpg(root='../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/',
                           csv_path='./splits.csv', mode='base_validate', resize=224)
train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for k, (data_inputs, data_labels) in enumerate(train_loader):
        inputs, labels = data_inputs.to(device), data_labels.to(device)
        # Set gradient to 0.
        optimizer.zero_grad()
        # Feed forward.
        pred = model(inputs)
        # Loss calculation.
        loss = criterion(pred, labels)
        # Gradient calculation.
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for k, (data_inputs, data_labels) in enumerate(test_loader):
            inputs, labels = data_inputs.to(device), data_labels.to(device)
            pred = model(inputs)
            loss = criterion(pred, labels)
            val_loss += loss.item()

    print(f'[{epoch + 1}] loss: {train_loss / k}')