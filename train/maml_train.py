import torch, os
import pandas as pd
import numpy as np
import scipy.stats
from torch.utils.data import DataLoader
import sys

sys.path.append('..')
from shared.datasets import *
from shared.meta import *


def main(k_shot):
    # Set seed for reproducibility
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    n_way = 3
    k_query = 16
    num_workers = 12
    train_num_episodes = 50000
    test_num_episodes = 200
    bs = 1
    root = '../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files'
    path_splits = '../splits/splits.csv'  # Location of preprocessed splits
    path_results = f'../../results/{k_shot}shot'  # Folder to save the CSV results

    # Create results folder if it does not exist
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    update_lr = 1e-2  # Learning rate for meta-training
    meta_lr = 1e-3  # Learning rate for meta-testing
    update_step = 5  # Number of meta-training update steps
    update_step_test = 10  # Number of meta-testing update steps
    imgsz = 224  # Size of images
    imgc = 1  # Initial image channels

    # Learner model configuration
    config = [
        ('conv2d', [64, 1, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('conv2d', [64, 64, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('max_pool2d', [2, 2, 0]),
        ('flatten', []),
        ('linear', [n_way, 64 * 14 * 14])
    ]

    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    maml = Meta(update_lr, meta_lr, n_way, k_shot, k_query, bs,
                update_step, update_step_test, imgc, imgsz, config).to(device)
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))

    # Create batched episode datasets
    mini = MimicCxrJpgEpisodes(root, path_splits, n_way, k_shot, k_query, train_num_episodes, mode="base")
    mini_test = MimicCxrJpgEpisodes(root, path_splits, n_way, k_shot, k_query, test_num_episodes, mode="novel")

    # fetch meta_batchsz num of episode each time
    db = DataLoader(mini, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=True)

    # Keep track of best meta-testing results
    best_score = 0
    best_step = 0

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(db):

        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

        accs = maml(x_spt, y_spt, x_qry, y_qry)

        if (step + 1) % 1000 == 0:  # evaluation
            # Create Dataframe containing results of the multiple episodes
            df_results = pd.DataFrame(columns=['Step', 'Accuracy', 'Macro Accuracy',
                                               'Macro-F1 Score'] + [str(x) + ' F1' for x in range(n_way)])
            db_test = DataLoader(mini_test, 1, shuffle=True, num_workers=1, pin_memory=True)
            accs_all_test = []

            for x_spt, y_spt, x_qry, y_qry in db_test:
                x_spt, y_spt = x_spt.squeeze(0).to(device), y_spt.squeeze(0).to(device)
                x_qry, y_qry = x_qry.squeeze(0).to(device), y_qry.squeeze(0).to(device)

                # Record the best step per episode
                df_best = maml.finetunning(x_spt, y_spt, x_qry, y_qry)
                df_results = df_results.append(df_best.loc[0], ignore_index=True)

            # Find average accuracy and average f1 score over the experiments
            average_accuracy = df_results["Accuracy"].mean()
            average_f1 = df_results["Macro-F1 Score"].mean()
            print(f'Step: {step} Accuracy: {average_accuracy} F1-Score: {average_f1}')  # Print results

            # Record best testing results
            score = 0.5 * average_accuracy + 0.5 * average_f1
            if score > best_score:
                best_score = score
                best_step = step
                df_best_test = df_results
                # Save consistently due to long training time
                df_best_test.to_csv(os.path.join(path_results, f'{k_shot}shot_MAML_{best_step}.csv'), index=False)

    print(f"Best Step: {best_step}")


if __name__ == '__main__':
    print(f'MAML Training {sys.argv[1]} shot')
    main(int(sys.argv[1]))  # Get the k_shot variable from command line
