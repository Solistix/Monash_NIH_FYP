from fastai import *
from fastai.vision import *
from fastai.callbacks import *
import sys
sys.path.append('../..')
from scripts.data_scripts import *


# Load in base class data
data = load_data('base')

# Create Learner
num_sample = [2592, 2850, 388, 1926, 328, 4327, 4000, 762, 823, 1860]
max_sample = max(num_sample)
weight = torch.FloatTensor([max_sample / x for x in num_sample]).cuda()

arch = models.densenet161
learn = cnn_learner(data, arch, metrics=[error_rate, accuracy, AUROC()],
                    loss_func=torch.nn.CrossEntropyLoss(weight=weight), callback_fns=[CSVLogger])

# Load model and get learning rate curve
model_path = '../../../../../home/ilu3/rl80/Monash_NIH_FYP/100_shot/baseline/stage1'
learn = learn.load(model_path)
learn.unfreeze()

learn.lr_find()
fig = learn.recorder.plot(return_fig=True)
fig.savefig('./stage1_lr.jpg')

