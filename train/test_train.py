import torch
import numpy as np
import pandas as pd
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from biobertology import get_tokenizer
import sys

sys.path.append('..')
from shared.models import *
from shared.datasets import *
from shared.metrics import *


def train(text_inputs, attention_inputs, labels, model, criterion, device, optimizer, freeze=False):
    # Training loop
    model.train()

    # Freeze all layers except those indicated
    if freeze:
        for name, param in model.named_parameters():
            if name not in freeze:
                param.requires_grad = False

    # Train the entire support set in one batch
    optimizer.zero_grad()
    pred = model(text_inputs, attention_inputs)
    loss = criterion(pred, labels)
    loss.backward()
    optimizer.step()
    train_loss = loss.item()  # Running training loss

    return train_loss


def test(text_inputs, attention_inputs, labels, model, criterion, device, n_way):
    # An F1 Score of 0 indicates that it is invalid
    model.eval()
    true_positive = list(0. for i in range(n_way))  # Number of correctly predicted samples per class
    total_truth = list(0. for i in range(n_way))  # Number of ground truths per class
    predicted_positive = list(0. for i in range(n_way))  # Number of predicted samples per class
    correct_total = 0  # Total correctly predicted samples
    total = 0  # Total samples
    with torch.no_grad():
        # Test the entire query set in one batch
        pred = model(text_inputs, attention_inputs)
        loss = criterion(pred, labels)
        val_loss = loss.item()  # Running validation loss
        _, predicted = torch.max(pred, 1)
        correct = (predicted == labels).squeeze()  # Samples that are correctly predicted
        correct_total += (predicted == labels).sum().item()
        total += labels.size(0)

        for i in range(len(predicted)):
            label = labels[i]
            true_positive[label] += correct[i].item()
            total_truth[label] += 1
            predicted_positive[predicted[i].item()] += 1  # True Positive + False Positive

    accuracy, macro_accuracy, f1_score, class_f1 = metrics(true_positive, total_truth,
                                                           predicted_positive, correct_total, total)

    return val_loss, accuracy, macro_accuracy, f1_score, class_f1


def main(k_shot):
    # Set seed for reproducibility
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    # Set Training Parameters
    n_way = 3
    k_query = 16
    num_episodes = 200
    num_epochs = 100
    num_workers = 12
    bs = 4
    lr = 1e-3
    root_text = '../../../../scratch/rl80/mimic-cxr-2.0.0.physionet.org'
    path_biobert = '../results'
    path_splits = '../splits/splits.csv'  # Location of preprocessed splits
    path_results = f'../../results/{k_shot}shot'  # Folder to save the CSV results
    freeze = ['linear.weight', 'linear.bias']  # Freeze all layers except linear layers

    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training tools
    criterion = nn.CrossEntropyLoss()
    tokenizer = get_tokenizer()

    print('Begin Loading')
    # Load in data
    dataset = MimicCxrReportsEpisodes(root_text, path_splits, tokenizer, n_way, k_shot, k_query, num_episodes, 'novel')
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

    # Create Dataframe to export results to CSV
    df_results = pd.DataFrame(columns=['Epoch', 'Training Loss', 'Validation Loss', 'Accuracy', 'Macro Accuracy',
                                       'Macro-F1 Score'] + [str(x) + ' F1' for x in range(n_way)])
    print('Finished Loading, Begin Training')
    # Iterate through batched episodes. One episode is one experiment
    for step, (support_texts, support_masks, support_labels, query_texts, query_masks, query_labels) in enumerate(loader):
        print('Loaded Tensors')
        # Convert Tensors to appropriate device
        batch_support_x, batch_support_masks, batch_support_y, batch_query_x, batch_query_masks, batch_query_y = \
            support_texts.to(device), support_masks.to(device), support_labels.to(device), \
            query_texts.to(device), query_masks.to(device), query_labels.to(device)
        print('Converted Tensors')
        # [num_batch, training_sz, channels, height, width] = support_x.size()
        # num_batch = num of episodes
        # training_sz = size of support or query set
        num_batch = batch_support_x.size(0)  # Number of episodes in the batch

        # Break down the batch of episodes into single episodes
        for i in range(num_batch):
            # Load in model and reset weights every episode/experiment
            print('Start Loading Model')
            model = SemanticNet(n_way, path_biobert).to(device)
            print('Finished Loading Model')

            # Reset optimizer with model parameters
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # Break down the sets into individual episodes
            support_x, support_m, support_y, query_x, query_m, query_y = \
                batch_support_x[i], batch_support_masks[i], batch_support_y[i], \
                batch_query_x[i], batch_query_masks[i], batch_query_y[i]

            # Variables for best epoch per experiment
            best_score = 0
            best_epoch = 0
            df_best = pd.DataFrame(columns=['Epoch', 'Training Loss', 'Validation Loss', 'Accuracy', 'Macro Accuracy',
                                            'Macro-F1 Score'] + [str(x) + ' F1' for x in
                                                                 range(n_way)])  # Track best epoch
            # Training and testing for specified epochs
            for epoch in range(num_epochs):
                # Training
                train_loss = train(support_x, support_m, support_y, model, criterion, device, optimizer, freeze=freeze)

                # Testing
                val_loss, acc, m_acc, macro_f1, class_f1 = test(query_x, query_m, query_y,
                                                                model, criterion, device, n_way)

                if step == 0 and i == 0:
                    print(
                        f'[{epoch + 1}] t_loss: {train_loss:.5f} v_loss: {val_loss:.5f} val_acc: {acc:.5f} '
                        f'val_m_acc: {m_acc:.5f} f1: {macro_f1:.5f}')
                # Find best epoch
                score = 0.5 * acc + 0.5 * macro_f1
                if score > best_score:
                    best_score = score
                    df_best.loc[0] = [epoch + 1, train_loss, val_loss, acc, m_acc, macro_f1] + class_f1

            # Print the best results per experiment
            print(
                f'[{int(df_best.iloc[0, 0])}] t_loss: {df_best.iloc[0, 1]} v_loss: {df_best.iloc[0, 2]} '
                f'val_acc: {df_best.iloc[0, 3]} f1: {df_best.iloc[0, 5]}')

            # Record the best epoch to be saved into a CSV
            df_results = df_results.append(df_best.loc[0], ignore_index=True)

    # Create results folder if it does not exist
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # Export results to a CSV file
    df_results.to_csv(os.path.join(path_results, f'{k_shot}shot_semantic.csv'), index=False)


if __name__ == '__main__':
    #print(f'Semantic Training {sys.argv[1]} shot')
    #main(int(sys.argv[1]))  # Get the k_shot variable from command line
    main(1)

