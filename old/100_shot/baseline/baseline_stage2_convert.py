from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import sys
sys.path.append('../..')
from scripts.data_scripts import *

# This script converts the output of the last linear layer from 10 to 3 and saves the model.

# Load in base class data
data = load_data('base')

# Create Learner
num_sample = [2592, 2850, 388, 1926, 328, 4327, 4000, 762, 823, 1860]
max_sample = max(num_sample)
weight = torch.FloatTensor([max_sample / x for x in num_sample]).cuda()

arch = models.densenet161
learn = cnn_learner(data, arch, metrics=[error_rate, accuracy, AUROC()],
                    loss_func=torch.nn.CrossEntropyLoss(weight=weight), callback_fns=[CSVLogger])

# Load Model and convert last layer from 10 classes to 3
model_path = '../../../../../home/ilu3/rl80/Monash_NIH_FYP/100_shot/baseline/stage2'
learn = learn.load(model_path)
learn.model[-1][-1] = nn.Linear(512, 3)
learn.save('stage2_class_10to3')
