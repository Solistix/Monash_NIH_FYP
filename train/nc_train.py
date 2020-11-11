"""
This file trains the Nearest Centroid model using the meta-learning framework. The feature extractor used is from the
basic model.
"""

import torch
import numpy as np
import sys
from baseline_train import *


def main(k_shot):
    # Set seed for reproducibility
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    n_way = 3
    k_query = 16
    num_episodes = 200
    num_workers = 12
    bs = 4
    root = '../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files'
    path_splits = '../splits/splits.csv'  # Location of preprocessed splits
    path_results = f'../../results/{k_shot}shot'  # Folder to save the CSV results
    path_pretrained = '../results/basic/basic_39.pth'

    # Set device to GPU if it exists
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # Load in data
    dataset = MimicCxrJpgEpisodes(root, path_splits, n_way, k_shot, k_query, num_episodes, 'novel')
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=num_workers)

    # Create Dataframe to export results to CSV
    df_results = pd.DataFrame(columns=['Validation Loss', 'Accuracy', 'Macro Accuracy',
                                       'Macro-F1 Score'] + [str(x) + ' F1' for x in range(n_way)])
    df_hold = pd.DataFrame(columns=['Validation Loss', 'Accuracy', 'Macro Accuracy',
                                    'Macro-F1 Score'] + [str(x) + ' F1' for x in range(n_way)])

    # Iterate through batched episodes. One episode is one experiment
    for step, (support_imgs, support_labels, query_imgs, query_labels) in enumerate(loader):
        # Convert Tensors to appropriate device
        batch_support_x, batch_support_y = support_imgs.to(device), support_labels.to(device)
        batch_query_x, batch_query_y = query_imgs.to(device), query_labels.to(device)

        # [num_batch, training_sz, channels, height, width] = support_x.size()
        # num_batch = num of episodes
        # training_sz = size of support or query set
        num_batch = batch_support_x.size(0)  # Number of episodes in the batch

        # Break down the batch of episodes into single episodes
        for i in range(num_batch):
            # Load in model and reset weights every episode/experiment
            model = CosineSimilarityNet(n_way).to(device)
            pretrained_dict = torch.load(path_pretrained)
            del pretrained_dict['linear.weight']  # Remove the last layer
            del pretrained_dict['linear.bias']
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

            # Break down the sets into individual episodes
            support_x, support_y = batch_support_x[i], batch_support_y[i]
            query_x, query_y = batch_query_x[i], batch_query_y[i]

            # Find Average Features
            model.eval()
            with torch.no_grad():
                # Initialise list containing features sorted by class
                label_features = [torch.FloatTensor([]).to(device) for i in range(n_way)]

                # Initialise weight for the nearest centroid, last layer weight
                fc_weight = torch.FloatTensor([]).to(device)

                # Get Features
                _, features = model(support_x, extract_features=True)

                # Sort features by labels to be averaged later on
                for i in range(features.size(0)):
                    label = support_y[i]
                    label_features[label] = torch.cat((label_features[label], features[i][None]))

                # Create weight for the last layer
                for j in range(n_way):
                    feature_avg = torch.mean(label_features[j], 0)
                    fc_weight = torch.cat((fc_weight, feature_avg[None]), 0)

                # Apply weight to the model
                nc_dict = model.state_dict()
                nc_dict['cos_sim.weight'] = fc_weight
                model.load_state_dict(nc_dict)

            # Testing
            val_loss, acc, m_acc, macro_f1, class_f1 = test(query_x, query_y, model, criterion, device, n_way)

            # Print the results per experiment
            print(f'[v_loss: {val_loss:.5f} val_acc: {acc:.5f} val_m_acc: {m_acc:.5f} f1: {macro_f1:.5f}')

            # Record the experiment to be saved into a CSV
            df_hold.loc[0] = [val_loss, acc, m_acc, macro_f1] + class_f1
            df_results = df_results.append(df_hold.loc[0], ignore_index=True)

    # Create results folder if it does not exist
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    df_results.to_csv(os.path.join(path_results, f'{k_shot}shot_nc.csv'), index=False)  # Export results to a CSV file


if __name__ == '__main__':
    print(f'NC Training {sys.argv[1]} shot')
    main(int(sys.argv[1]))  # Get the k_shot variable from command line
