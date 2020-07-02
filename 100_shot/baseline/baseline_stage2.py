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

# Load Model and Unfreeze parameters for fine-tuning
learn.load('baseline_model_stage1/bestmodel_3') # IMPORTANT NEED TO SELECT MODEL
learn.unfreeze()

learn.model = learn.model.cuda() # IMPORTANT NEED TO SELECT LR
learn.fit_one_cycle(10, max_lr=slice(lr, 3e-4), callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')])
