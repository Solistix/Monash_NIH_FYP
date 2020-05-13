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

data = (ImageList.from_df(df, path_jpg)
        .split_by_idxs(train_idx, valid_idx)
        .label_from_df(label_delim=',')
        .transform('', size=224)
        .databunch(bs=64)
        .normalize(imagenet_stats))


# In[11]:


learn = cnn_learner(data, models.densenet121, metrics=[error_rate, accuracy], callback_fns=[CSVLogger])


# In[6]:


# learn.lr_find()
# learn.recorder.plot()


# In[7]:
print('normal')
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())
lr = 0.01
learn.fit_one_cycle(100, slice(lr), callbacks=[SaveModelCallback(learn, every='epoch', monitor='accuracy')])

