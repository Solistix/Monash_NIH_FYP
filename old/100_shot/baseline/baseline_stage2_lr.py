from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import sys
sys.path.append('../..')
from scripts.data_scripts import *


# Load in base class data
data = load_data('novel')

# Create Learner
arch = models.densenet161
learn = cnn_learner(data, arch, metrics=[error_rate, accuracy, AUROC()],
                    loss_func=torch.nn.CrossEntropyLoss(), callback_fns=[CSVLogger])

# Load model and get learning rate curve
model_path = '../../../../../home/ilu3/rl80/Monash_NIH_FYP/100_shot/baseline/stage2_class_10to3'
learn = learn.load(model_path)

# Freeze every layer except the last
for group in learn.layer_groups:
    for layer in group:
        requires_grad(layer, False)

requires_grad(learn.layer_groups[-1][-1], True)

learn.lr_find()
fig = learn.recorder.plot(return_fig=True)
fig.savefig('./stage2_lr.jpg')
