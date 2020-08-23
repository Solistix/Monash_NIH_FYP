import torch
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms


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
