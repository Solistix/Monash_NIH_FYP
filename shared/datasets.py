import torch
import pandas as pd
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pathlib import Path


class MimicCxrJpg(Dataset):
    """
    Mimic-CXR-JPG Database
    Todo: Insert references to the database here!
    """

    def __init__(self, root, path_csv, mode, resize=224):

        # Check if mode contains an accepted value
        if mode not in ('base_train', 'base_validate'):
            raise Exception("Selected 'mode' is not valid")

        self.root = root
        csv_data = pd.read_csv(path_csv)
        self.data = csv_data[csv_data.split == mode]
        self.resize = resize
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('L'),
                                             transforms.Resize((self.resize, self.resize)),
                                             transforms.ToTensor()
                                             ])

        self.dict_labels = {
            'Atelectasis': 0,
            'Cardiomegaly': 1,
            'Consolidation': 2,
            'Edema': 3,
            'No Finding': 4,
            'Pneumonia': 5,
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root, self.data.iloc[idx, 0])
        img_tensor = self.transform(img_path)
        label = self.data.iloc[idx, 1]

        return img_tensor, self.dict_labels[label]


class MimicCxrJpgEpisodes(Dataset):
    """
    Mimic-CXR-JPG Database
    Todo: Insert references to the database here!
    """

    def __init__(self, root, path_csv, n_way, k_shot, k_query, num_episodes, mode, resize=224):

        # Check if mode contains an accepted value
        if mode not in ('base', 'novel'):
            raise Exception("Selected 'mode' is not valid")

        self.root = root
        csv_data = pd.read_csv(path_csv)  # Raw CSV data

        if mode == 'base':
            self.dict_labels = {
                'Atelectasis': 0,
                'Cardiomegaly': 1,
                'Consolidation': 2,
                'Edema': 3,
                'No Finding': 4,
                'Pneumonia': 5,
            }
            # Filters for novel classes
            data = csv_data[(csv_data.split == "base_train") | (csv_data.split == "base_validate")]

        else:
            self.dict_labels = {
                'Enlarged Cardiomediastinum': 0,
                'Fracture': 1,
                'Lung Lesion': 2,
                'Lung Opacity': 3,
                'Pleural Effusion': 4,
                'Pneumothorax': 5
            }
            data = csv_data[csv_data.split == "novel"]  # Filters for novel classes

        self.data = data.assign(
            labels=data["labels"].apply(lambda x: self.dict_labels[x]))  # Converts classes to numeric values
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.num_episodes = num_episodes
        self.resize = resize
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('L'),
                                             transforms.Resize((self.resize, self.resize)),
                                             transforms.ToTensor()
                                             ])

        # Create Episodes
        self.support_episodes = []  # List of training episodes (support set)
        self.query_episodes = []  # List of testing episodes (query set)
        for i in range(self.num_episodes):  # for each batch
            # 1.select n_way classes randomly
            selected_cls = np.random.choice(len(self.dict_labels), self.n_way, False)  # no duplicate
            np.random.shuffle(selected_cls)
            df_support = pd.DataFrame()
            df_query = pd.DataFrame()
            for cls in selected_cls:
                df_cls = self.data[self.data.labels == cls]
                # 2. select k_shot + k_query for each class
                selected_idx = np.random.choice(len(df_cls), self.k_shot + self.k_query, False)
                np.random.shuffle(selected_idx)

                # Index of samples for the support and query set
                support_idx = selected_idx[:self.k_shot]
                query_idx = selected_idx[self.k_shot:]

                df_support = df_support.append(df_cls.iloc[support_idx])
                df_query = df_query.append(df_cls.iloc[query_idx])

            # Shuffle the indexes so that it is no longer ordered by class
            df_support = df_support.sample(frac=1)
            df_query = df_query.sample(frac=1)

            self.support_episodes.append(df_support)
            self.query_episodes.append(df_query)

    def __len__(self):
        return self.num_episodes

    def __getitem__(self, idx):
        support_set = self.support_episodes[idx]
        query_set = self.query_episodes[idx]

        # Labels ranging from 0 to (number of classes -1)
        support_labels = support_set.labels.tolist()
        query_labels = query_set.labels.tolist()

        # Convert labels to range from 0 to (n way-1) for loss calculation
        unique_labels = np.unique(support_labels)  # Unique labels are the same for support and query set
        converted_support_labels = support_labels
        converted_query_labels = query_labels
        for idx, val in enumerate(unique_labels):
            # Get indexes of labels that are equal to the iterated val
            idx_support = [x for x, label in enumerate(support_labels) if label == val]
            idx_query = [x for x, label in enumerate(query_labels) if label == val]

            # Replace old labels with new labels
            for idx_change in range(len(idx_support)):
                converted_support_labels[idx_support[idx_change]] = idx

            for idx_change in range(len(idx_query)):
                converted_query_labels[idx_query[idx_change]] = idx

        support_imgs = torch.Tensor()
        for i in range(len(support_set)):
            img_path = os.path.join(self.root, support_set.iloc[i, 0])
            support_imgs = torch.cat((support_imgs, self.transform(img_path)[None]))  # 'None' index to add channel dim

        query_imgs = torch.Tensor()
        for j in range(len(query_set)):
            img_path = os.path.join(self.root, query_set.iloc[j, 0])
            query_imgs = torch.cat((query_imgs, self.transform(img_path)[None]))  # 'None' index to add channel dim

        return support_imgs, torch.LongTensor(support_labels), query_imgs, torch.LongTensor(query_labels)