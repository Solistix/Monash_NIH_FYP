import torch
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from pathlib import Path


class MimicCxrJpg(Dataset):
    """
    Mimic-CXR-JPG Database
    Todo: Insert references to the database here!
    """

    def __init__(self, root, csv_path, mode, resize):

        # Check if mode contains an accepted value
        if mode not in ('base_train', 'base_validate', 'novel_train', 'novel_validate'):
            raise Exception("Selected 'mode' is not valid")

        self.root = root
        csvdata = pd.read_csv(csv_path)
        self.data = csvdata[csvdata.split == mode]
        self.resize = resize
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('L'),
                                             transforms.Resize((self.resize, self.resize)),
                                             transforms.ToTensor()
                                             ])

        if mode == 'base_train' or mode == 'base_validate':
            self.dict_labels = {
                'Atelectasis': 0,
                'Cardiomegaly': 1,
                'Consolidation': 2,
                'Edema': 3,
                'Fracture': 4,
                'Lung Opacity': 5,
                'No Finding': 6,
                'Pneumonia': 7,
                'Pneumothorax': 8,
                'Support Devices': 9
            }
        else:
            self.dict_labels = {
                'Enlarged Cardiomediastinum': 0,
                'Lung Lesion': 1,
                'Pleural Effusion': 2,
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


class MimicCxrMulti(Dataset):
    """
    MIMIC-CXR-JPG Images and MIMIC-CXR Reports
    Todo: Insert references to the database here!
    Removes '_' from reports
    Truncates the reports to 512 tokens by removing the beginning of the report (Usually where the 'wet read' resides)
    """

    def __init__(self, root_image, root_text, csv_path, tokenizer, mode, resize=224, max_length=512):

        # Check if mode contains an accepted value
        if mode not in ('base_train', 'base_validate', 'novel_train', 'novel_validate'):
            raise Exception("Selected 'mode' is not valid")

        # Initialise variables
        self.root_text = root_text
        self.root_image = root_image
        self.resize = resize
        self.max_length = max_length
        self.transform = transforms.Compose([lambda x: Image.open(x).convert('L'),  # Transforms for images
                                             transforms.Resize((self.resize, self.resize)),
                                             transforms.ToTensor()
                                             ])
        self.tokenizer = tokenizer

        # Load data
        csv_data = pd.read_csv(csv_path)
        self.data = csv_data[csv_data.split == mode]

        if mode == 'base_train' or mode == 'base_validate':
            self.dict_labels = {
                'Atelectasis': 0,
                'Cardiomegaly': 1,
                'Consolidation': 2,
                'Edema': 3,
                'Fracture': 4,
                'Lung Opacity': 5,
                'No Finding': 6,
                'Pneumonia': 7,
                'Pneumothorax': 8,
                'Support Devices': 9
            }
        else:
            self.dict_labels = {
                'Enlarged Cardiomediastinum': 0,
                'Lung Lesion': 1,
                'Pleural Effusion': 2,
            }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Extract CSV data
        file_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]

        # Get image tensor
        img_path = os.path.join(self.root_image, file_path)  # Absolute file path to the JPG img
        img_tensor = self.transform(img_path)

        # Get text tensor and attention mask
        text_name = f'{file_path.split("/")[2]}.txt'  # Extract the study id to find the report
        text_path = Path(os.path.join(self.root_text, text_name))
        plain_text = text_path.read_text()
        plain_text = plain_text.replace('_', '')  # Remove all underscores from the text
        encoded_text = self.tokenizer.encode(plain_text, add_special_tokens=True)

        # Transform encodings to be of the same size
        len_encoding = len(encoded_text)
        if len_encoding > self.max_length:
            # Truncate to max length
            cutoff = len_encoding - self.max_length + 1  # The cutoff for the tokens to be deleted
            del encoded_text[1:cutoff]
            attention = [1] * self.max_length
        elif len_encoding < self.max_length:
            # Pad to max length
            num_padding = self.max_length - len_encoding
            encoded_text.extend([0] * num_padding)  # Padding token is 0
            attention = [1] * len_encoding
            attention.extend([0] * (self.max_length - len_encoding))
        else:
            # If equal size, create attention matrix
            attention = [1] * self.max_length

        text_tensor = torch.tensor(encoded_text)
        attention_tensor = torch.tensor(attention)

        return img_tensor, text_tensor, attention_tensor, self.dict_labels[label]


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from torchvision.utils import make_grid
    from torch.utils.data import DataLoader

    mimic_dataset = MimicCxrJpg(root='../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/',
                                csv_path='./splits.csv', mode='base_train', resize=224)

    dataloader = DataLoader(mimic_dataset, batch_size=4,
                            shuffle=True, num_workers=0)

    for i_batch, batch in enumerate(dataloader):
        plt.figure()
        image_batch = batch[0]
        plt.figure()
        grid = make_grid(image_batch, nrow=2)
        plt.imshow(grid.permute(1, 2, 0), cmap='gray')

        # Show two batches
        if i_batch == 1:
            plt.show()
            break
