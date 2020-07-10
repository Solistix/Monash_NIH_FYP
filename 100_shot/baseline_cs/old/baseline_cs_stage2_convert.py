from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import sys
sys.path.append('../..')
from scripts.data_scripts import *
from scripts.layers import *


# Load in base class data
data = load_data('base')

# Use custom header to replace classifier layers
custom_head = create_head(4416, 2, lin_ftrs=[512])
custom_head[-1] = CosineSimilarityLayer(512, 10)

# Create Learner
num_sample = [2592, 2850, 388, 1926, 328, 4327, 4000, 762, 823, 1860]
max_sample = max(num_sample)
weight = torch.FloatTensor([max_sample / x for x in num_sample]).cuda()

arch = models.densenet161
learn = cnn_learner(data, arch, metrics=[error_rate, accuracy],
                    loss_func=torch.nn.CrossEntropyLoss(weight=weight),
                    callback_fns=[CSVLogger],
                    custom_head=custom_head)

# Load Model and convert last layer from 10 classes to 3
model_path = '../../../../../home/ilu3/rl80/Monash_NIH_FYP/100_shot/baseline_cs/models/stage2'
learn = learn.load(model_path)
learn.model[-1][-1] = CosineSimilarityLayer(512, 3)
learn.save('stage2_class_10to3')
