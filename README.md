# Summary
The models in this git repository are trained on a meta-learning framework. This involves training on multiple few-shot tasks. 
A few-shot task is defined as a N-way K-shot problem where N is the number of classes in that task and K is the number of samples in the support set per class. Q will also be defined as the number of query samples per set. In this repository we will use “task” and “episodes” interchangeably.

# Datasets
The data used in the training of the models is from the [MIMIC-CXR]( https://physionet.org/content/mimic-cxr/2.0.0/) and [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr-jpg/2.0.0/). 

The reports used are from the MIMIC-CXR while the labels and chest X-ray images are from the MIMIC-CXR-JPG. 

The file structure used in this project has the reports downloaded from the MIMIC-CXR all in one folder. Only the reports are required from this dataset. MIMIC-CXR-JPG has the same file structure as the entire dataset.
# Usage
The pre-generated splits of the data are already provided but if changes want to be made, this can be done by running the “create_splits.py” file in the scripts folder.

The BioBERT weights can also be downloaded using the download_biobert.py script in the same folder.

# Model Summary
The training scripts for each model can be found in the train folder.

## Feature Extractors
The image feature extractors are made from the basic and basic cosine models. The basic model is used in the training of the baseline, NC (Nearest Centroid) and the Multi Modal model.

The basic cosine is used in the training of the “cosine_similarity” (Baseline with cosine distance function) and NC_CS (Nearest Centroid).

A pre-trained BioBERT model is used as the feature extractor for the semantic and multi-modal model. Information about the BioBERT model can be obtained [here](https://arxiv.org/abs/1901.08746). The BioBERT model was implemented using this package ()[https://pypi.org/project/biobert-pytorch/]

Feature extractors are trained on normal supervised learning.

## Few-Shot Models
Baseline - Linear is based on the baseline given in this [paper](https://openreview.net/pdf?id=HkxLXnAcFQ). This model is trained on baseline_train.py.

Baseline – Cosine is based on the Baseline++ of the above paper and is trained on cosine_similarity_train.py.

Nearest Centroid – Linear is based on the classifier baseline in this [paper](https://arxiv.org/pdf/2003.04390.pdf).  It is named linear as it is trained using the basic model as the feature extractor.

Nearest Centroid – Cosine is the same model as above except it is trained using the basic cosine model as the feature extractor.

MAML is based on the MAML algorithm in this [paper](https://arxiv.org/abs/1703.03400). The code in this repository for the implementation of this was adapted from [this](https://github.com/dragen1860/MAML-Pytorch).

All these models are trained on the meta-learning framework.

## Other Models
The semantic model is the BioBERT feature extractor combined with a linear classification layer so that it can serve as a point of comparison to the multi-modal model.

The multi-modal model is a simple concatenation of the basic and BioBERT features fed into a linear classification layer.
