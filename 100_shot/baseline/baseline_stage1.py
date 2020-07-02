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

learn.model = learn.model.cuda()
learn.fit_one_cycle(10, callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')])
