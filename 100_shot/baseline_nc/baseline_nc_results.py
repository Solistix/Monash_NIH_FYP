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
learn = cnn_learner(data, arch, metrics=[accuracy],
                    loss_func=torch.nn.CrossEntropyLoss(),
                    custom_head=custom_head)

# Load model
model_path = '../../../../../home/ilu3/rl80/Monash_NIH_FYP/100_shot/baseline_nc/stage3'
learn = learn.load(model_path, strict=False) # Need strict=False to ignore the extra bias parameters

# Results
results = learn.validate(data.valid_dl)
with open('history_stage3.txt', 'w') as f:
    print('Accuracy:', round(results[1].item(), 4), file=f)
