import torch.optim as optim
from torch.utils.data import DataLoader
import sys

sys.path.append('..')
from shared.models import *
from shared.datasets import *


def train(model, train_loader, criterion, device, optimizer):
    model.train()
    train_loss = 0
    for step, (data_inputs, data_labels) in enumerate(train_loader):
        inputs, labels = data_inputs.to(device), data_labels.to(device)  # Convert Tensors to appropriate device
        optimizer.zero_grad()
        pred = model(inputs)
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()  # Running training loss

    return train_loss / step


def test(model, test_loader, criterion, device, n_way):
    model.eval()
    true_positive = list(0. for i in range(n_way))  # Number of correctly predicted samples per class
    total_truth = list(0. for i in range(n_way))  # Number of ground truths per class
    predicted_positive = list(0. for i in range(n_way))  # Number of predicted samples per class
    precision = list(0. for i in range(n_way))
    recall = list(0. for i in range(n_way))
    val_loss = 0
    correct_total = 0  # Total correctly predicted samples
    total = 0  # Total samples
    f1_flag = 0  # Flag for if the model does not predict any positives for a class which breaks precision and F1 score
    with torch.no_grad():
        for step, (data_inputs, data_labels) in enumerate(test_loader):
            inputs, labels = data_inputs.to(device), data_labels.to(device)
            pred = model(inputs)
            loss = criterion(pred, labels)
            val_loss += loss.item()  # Running validation loss
            _, predicted = torch.max(pred, 1)
            correct = (predicted == labels).squeeze()  # Samples that are correctly predicted
            correct_total += (predicted == labels).sum().item()
            total += labels.size(0)

            for i in range(len(predicted)):
                label = labels[i]
                true_positive[label] += correct[i].item()
                total_truth[label] += 1
                predicted_positive[predicted[i].item()] += 1  # True Positive + False Positive

        # Find class accuracy, precision and recall
        for j in range(n_way):
            if (predicted_positive[j] == 0 or true_positive[j] == 0):
                f1_flag = 1
            else:
                precision[j] = true_positive[j] / predicted_positive[j]
            recall[j] = true_positive[j] / total_truth[j]  # Recall is the same as per class accuracy

        # Find Accuracy, Macro Accuracy and Macro F1 Score
        macro_acc_sum = 0
        f1_sum = 0
        for k in range(n_way):
            macro_acc_sum += recall[k]
            if f1_flag == 0:  # Check for broken f1 score
                f1_sum = 2 * precision[k] * recall[k] / (precision[k] + recall[k])

        accuracy = correct_total / total
        macro_accuracy = macro_acc_sum / n_way
        if f1_flag == 0:
            f1_score = f1_sum / n_way
        else:
            f1_score = -1

    return val_loss / step, accuracy, macro_accuracy, f1_score


if __name__ == '__main__':

    num_epochs = 30
    num_workers = 8
    bs = 64
    n_way = 10

    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BaselineNet(n_way).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_dataset = MimicCxrJpg(root='../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/',
                                csv_path='./splits.csv', mode='base_train', resize=224)
    test_dataset = MimicCxrJpg(root='../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/',
                               csv_path='./splits.csv', mode='base_validate', resize=224)
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, device, optimizer)
        val_loss, acc, m_acc, f1 = test(model, test_loader, criterion, device, n_way)

        print(
            f'[{epoch + 1}] t_loss: {train_loss:.5f} v_loss: {val_loss:.5f} '
            f'val_acc: {acc:.5f} val_m_acc: {m_acc:.5f} f1: {f1:.5f}')
