# In[1]:
# In[10]:


from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from pathlib import Path
import os
import torch
import matplotlib
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
size = 224
data = (ImageList.from_df(df, path_jpg)
        .split_by_idxs(train_idx, valid_idx)
        .label_from_df(cols='labels')
        .transform(tfms=tfms, size=size, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=64)
        .normalize(imagenet_stats))


# In[11]:


learn = cnn_learner(data, models.densenet121, metrics=[error_rate, accuracy], callback_fns=[CSVLogger])
learn.load('baseline_model_stage1/bestmodel_3')
learn.unfreeze()

# In[6]:


learn.lr_find()
fig = learn.recorder.plot(return_fig=True)
fig.savefig('LR_Curve.jpg')


# In[7]:
# lr = 0.01
# learn.fit_one_cycle(20, slice(lr), callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')])

