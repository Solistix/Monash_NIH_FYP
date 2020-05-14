# In[1]:
# In[10]:


from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from pathlib import Path
import os
import torch
print(torch.cuda.is_available())

# In[3]:


path_splits = '../splits/100_shot_split.csv'
path_jpg = Path('../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/')

df = pd.read_csv(path_splits)
data = ImageList.from_df(df, path_jpg)


# In[4]:


train_idx = df.index[df['split']=='train']
valid_idx = df.index[df['split']=='validate']
tfms = None
size = 448
bs = 16
data = (ImageList.from_df(df, path_jpg)
        .split_by_idxs(train_idx, valid_idx)
        .label_from_df(cols='labels')
        .transform(tfms=tfms, size=size, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=bs)
        .normalize(imagenet_stats))


# In[11]:

arch = models.densenet161
learn = cnn_learner(data, arch, metrics=[error_rate, accuracy], callback_fns=[CSVLogger])
#learn.load('baseline_model_stage1/bestmodel_1')
#learn.unfreeze()

# In[6]:


# learn.lr_find()
# learn.recorder.plot()


# In[7]:
#learn.fit_one_cycle(10, max_lr=slice(1e-5,1e-4), callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')])
learn.fit_one_cycle(20, callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')])

