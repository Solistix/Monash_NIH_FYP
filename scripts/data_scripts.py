from fastai.vision import *
from pathlib import Path


def load_data(class_type, tfms=None, size=224, bs = 16, path_splits='../splits.csv',
              path_jpg=Path('../../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org/files/')):
    df = pd.read_csv(path_splits)

    if class_type == 'base':
        train_idx = df.index[df['split'] == 'base_train']
        valid_idx = df.index[df['split'] == 'base_validate']
    elif class_type == 'novel':
        train_idx = df.index[df['split'] == 'novel_train']
        valid_idx = df.index[df['split'] == 'novel_validate']
    else:
        raise Exception("Invalid class type input")

    ret_data = (ImageList.from_df(df, path_jpg)
                .split_by_idxs(train_idx, valid_idx)
                .label_from_df(cols='labels')
                .transform(tfms=tfms, size=size, resize_method=ResizeMethod.SQUISH)
                .databunch(bs=bs)
                .normalize(imagenet_stats))

    return ret_data
