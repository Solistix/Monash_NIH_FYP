from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import sys
sys.path.append('../..')
from scripts.data_scripts import *
from scripts.layers import *

# Load in base class data
data = load_data('novel')

# Use custom header to replace classifier layers
custom_head = create_head(4416, 3, lin_ftrs=[512])
custom_head[-1] = CosineSimilarityLayer(512, 3)

# Create Learner
arch = models.densenet161
learn = cnn_learner(data, arch, metrics=[error_rate, accuracy],
                    loss_func=torch.nn.CrossEntropyLoss(),
                    custom_head=custom_head)

# Load model
model_path = '../../../../../home/ilu3/rl80/Monash_NIH_FYP/100_shot/baseline_cs/models/stage1_class_10to3'
learn = learn.load(model_path)

# Freeze every layer except the last
for group in learn.layer_groups:
    for layer in group:
        requires_grad(layer, False)

requires_grad(learn.layer_groups[-1][-1], True)

# Get Learning Rate
learn.lr_find()
fig = learn.recorder.plot(return_fig=True)
fig.savefig('./stage2_copy_lr.jpg')

