from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import sys
sys.path.append('../..')
from scripts.data_scripts import *
from scripts.layers import *


# Parameters for the datablock API
tfms = None
size = 224
bs = 16
path_splits = '../splits.csv'
path_jpg = Path('../../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/')

# Load in novel class data
data = load_data('novel')

# Create Learner
arch = models.densenet161
learn = cnn_learner(data, arch, metrics=[error_rate, accuracy], loss_func=torch.nn.CrossEntropyLoss())

# Load model
model_path = '../../../../../home/ilu3/rl80/Monash_NIH_FYP/models/baseline/stage2_class_10to3'
learn = learn.load(model_path)

# Get samples per class to create weights
# Get index for the relevant labels
df = pd.read_csv(path_splits)
train_idx = []
train_idx.append(df.index[(df['split'] == 'novel_train') & (df['labels'] == 'Enlarged Cardiomediastinum')])
train_idx.append(df.index[(df['split'] == 'novel_train') & (df['labels'] == 'Lung Lesion')])
train_idx.append(df.index[(df['split'] == 'novel_train') & (df['labels'] == 'Pleural Effusion')])

# Hook to get the output of the second last layer
def feature_hook(model, input, output):
    global activation
    activation = torch.cat((activation, output), 0)

hook = learn.model[-1][-2].register_forward_hook(feature_hook)

data_novel = []
weight = torch.FloatTensor([]).cuda()
for i in range(3):
    # Create a list containing only samples from each label
    data_novel.append((ImageList.from_df(df, path_jpg)
                       .split_by_idxs(train_idx[i], train_idx[i])
                       .label_from_df(cols='labels')
                       .transform(tfms=tfms, size=size, resize_method=ResizeMethod.SQUISH)
                       .databunch(bs=bs)
                       .normalize(imagenet_stats))
                      )

    # Get Activations
    activation = torch.FloatTensor([]).cuda()
    learn.validate(data_novel[i].valid_dl)  # Use validation set as the training set is a multiple of the batch size
    mean = torch.mean(activation, 0)
    weight = torch.cat((weight, mean[None]), 0)

# Replace the layer parameters with the new feature average
param = nn.Parameter(weight, True)
learn.layer_groups[-1][-1].weight = param
learn.save('stage3')
