import os
import pandas as pd
from pathlib import Path


def create_splits(path_metadata, path_splits):
    """
    Create curated splits for the MIMIC-CXR-JPG Database.

    The following transformation is applied to the database:
        Keeps only affirmative data,
        Merges the two set of structured labels
        Removes disagreeing samples and multi-class samples
        Removes the Pleural Other and Support Devices Class
        Keeps only Antero-posterior oriented samples
        Undersamples the No Finding class to 5000 samples
        Exports the splits into a csv file

    :param path_metadata: The relative path of the folder that the metadata, chexpert and negbio information is stored
    :type path_metadata: str
    :param path_splits: The relative path that the CSV file will be saved to
    :type path_splits: str
    :return:
    """
    novel_labels = ['Lung Lesion', 'Lung Opacity', 'Enlarged Cardiomediastinum',
                    'Pleural Effusion', 'Pneumothorax', 'Fracture']

    # Load in data
    path_chexpert = Path(os.path.join(path_metadata, 'mimic-cxr-2.0.0-chexpert.csv.gz'))
    path_negbio = Path(os.path.join(path_metadata, 'mimic-cxr-2.0.0-negbio.csv.gz'))
    path_metadata = Path(os.path.join(path_metadata, 'mimic-cxr-2.0.0-metadata.csv.gz'))

    df_chexpert = pd.read_csv(path_chexpert)
    df_negbio = pd.read_csv(path_negbio)
    df_metadata = pd.read_csv(path_metadata)

    # Merge relevant metadata, NegBio labels and Chexpert labels
    df = df_negbio.merge(
        df_chexpert,
        how='left',
        left_on=['subject_id', 'study_id'], right_on=['subject_id', 'study_id'],
        suffixes=('', '_cx')
    )

    df_metadata.drop([
        'PerformedProcedureStepDescription',
        'Rows',
        'Columns',
        'StudyDate',
        'StudyTime',
        'ProcedureCodeSequence_CodeMeaning',
        'ViewCodeSequence_CodeMeaning',
        'PatientOrientationCodeSequence_CodeMeaning'
    ], axis=1, inplace=True)

    df = df_metadata.merge(
        df,
        how='left',
        left_on=['subject_id', 'study_id'], right_on=['subject_id', 'study_id'],
    )

    # Preprocess data:
    # Only use data that is a '1.0'
    # Remove all disagreeing '1.0' data
    # Remove all Pleural Other and Support Devices findings
    # Remove all non antero-posterior (AP) data
    for key in df.columns:
        if key in ('dicom_id', 'subject_id', 'study_id', 'ViewPosition'):
            continue

        if key[-3:] == '_cx':
            continue

        # Remove data that is not a '1.0'
        df[key] = df[key].map({1: key})
        df[key + '_cx'] = df[key + '_cx'].map({1: key})

        # Remove all disagreeing '1.0' data
        agree_matrix = df[key].fillna(0) == df[key + '_cx'].fillna(0)
        df = df[agree_matrix]

    # Remove all Pleural Other Data
    keep = df['Pleural Other'].map({'Pleural Other': False}).fillna(True)
    df = df[keep]

    # Remove all Pleural Other Data
    keep = df['Support Devices'].map({'Support Devices': False}).fillna(True)
    df = df[keep]

    # Remove all non antero-posterior (AP) data
    keep = df['ViewPosition'].map({'AP': True}).fillna(False)
    df = df[keep]

    # Remove Columns
    df.drop([key for key in df.columns if key[-3:] == '_cx'], axis=1, inplace=True)
    df.drop(['ViewPosition', 'Pleural Other', 'Support Devices'], axis=1, inplace=True)

    # Separate columns into path and labels
    df_labels = df.copy()
    cols_path = [key for key in df.columns if key in ('dicom_id', 'subject_id', 'study_id')]
    cols_labels = [key for key in df.columns if key not in ('dicom_id', 'subject_id', 'study_id')]

    # Combine columns into a file path and labels
    df_labels['file_path'] = df_labels[cols_path].apply(
        lambda x: f"p{str(x.values[1])[:2]}/p{x.values[1]}/s{x.values[2]}/{x.values[0]}.jpg", axis=1)
    df_labels['labels'] = df_labels[cols_labels].apply(lambda x: ','.join(x.dropna().values.tolist()), axis=1)
    df_labels.drop(df.columns, axis=1, inplace=True)

    # Remove all data that does not have a label
    df_labels = df_labels[~(df_labels['labels'] == '')]

    # Remove all multi label data
    keep = df_labels['labels'].apply(lambda x: ',' not in x)
    df_single_labels = df_labels[keep]

    df_splits = df_single_labels.copy()

    # Create base training splits of: 80% Training and 20% Validation per class
    # Label novel classes to be used in episode generation
    for label in cols_labels:
        # Create a dataframe for the single label in this iteration
        df_unsplit = df_splits[df_splits['labels'].apply(lambda x: x == label)]

        # Base Classes
        if label not in novel_labels:
            # Undersample the 'No Finding' Class to 5000 samples
            if label == 'No Finding':
                df_unsplit = df_unsplit.sample(5000, random_state=1)

            df_train = df_unsplit.sample(frac=0.8, random_state=1)
            df_validate = df_unsplit.drop(df_train.index)

            # Give split designation and merge back into main dataframe
            df_train['split'] = 'base_train'
            df_validate['split'] = 'base_validate'
            df_train.drop(['file_path', 'labels'], axis=1, inplace=True)
            df_validate.drop(['file_path', 'labels'], axis=1, inplace=True)

            # Merge the training dataframe into the main dataframe
            df_splits = df_splits.merge(
                df_train,
                how='left',
                left_index=True,
                right_index=True,
                suffixes=('', '_x')
            )

            # Combines values of the split columns
            if 'split_x' in df_splits.columns:
                df_splits['split'] = df_splits[['split', 'split_x']].apply(
                    lambda x: ''.join(x.dropna().values.tolist()), axis=1)
                df_splits.drop('split_x', axis=1, inplace=True)

            # Merge the validation dataframe into the main dataframe
            df_splits = df_splits.merge(
                df_validate,
                how='left',
                left_index=True,
                right_index=True,
                suffixes=('', '_x')
            )

            # Combines values of the split columns
            if 'split_x' in df_splits.columns:
                df_splits['split'] = df_splits[['split', 'split_x']].apply(
                    lambda x: ''.join(x.dropna().values.tolist()), axis=1)
                df_splits.drop('split_x', axis=1, inplace=True)

        # Novel Classes
        else:
            # Label the entire class as 'novel'
            df_unsplit['split'] = 'novel'
            df_unsplit.drop(['file_path', 'labels'], axis=1, inplace=True)

            # Merge the novel dataframe into the main dataframe
            df_splits = df_splits.merge(
                df_unsplit,
                how='left',
                left_index=True,
                right_index=True,
                suffixes=('', '_x')
            )

            # Combines values of the split columns
            if 'split_x' in df_splits.columns:
                df_splits['split'] = df_splits[['split', 'split_x']].apply(
                    lambda x: ''.join(x.dropna().values.tolist()), axis=1)
                df_splits.drop('split_x', axis=1, inplace=True)

    # Create splits folder if it does not exist
    if not os.path.exists(path_splits):
        os.makedirs(path_splits)

    # Export CSV containing splits
    df_splits.to_csv(os.path.join(path_splits, 'splits.csv'), index=False)


def check_splits(path_csv):
    """
    Sums up the number of training, validation and novel samples per class

    :param path_csv: The path to the CSV containing the splits data
    :type path_csv: str
    :return: A tuple of dictionaries detailing the number of samples per class
    :rtype: tuple
    """
    df_splits = pd.read_csv(path_csv)
    dict_train = {}
    dict_validate = {}
    dict_novel = {}
    for index, row in df_splits.iterrows():
        # Counts the number of samples per class in the 'base_train' split category
        if row['split'] == 'base_train':
            if row['labels'] in dict_train.keys():
                dict_train[row['labels']] += 1
            else:
                dict_train[row['labels']] = 1

        # Counts the number of samples per class in the 'base_validate' split category
        elif row['split'] == 'base_validate':
            if row['labels'] in dict_validate.keys():
                dict_validate[row['labels']] += 1
            else:
                dict_validate[row['labels']] = 1

        # Counts the number of samples per class in the 'novel' split category
        elif row['split'] == 'novel':
            if row['labels'] in dict_novel.keys():
                dict_novel[row['labels']] += 1
            else:
                dict_novel[row['labels']] = 1

    return dict_train, dict_validate, dict_novel


if __name__ == '__main__':
    path_metadata = '../../../../scratch/rl80/mimic-cxr-jpg-2.0.0.physionet.org'  # Folder of the MIMIC-CXR-JPG metadata
    path_splits = '../splits'  # Folder to save the splits to

    # Create Splits
    create_splits(path_metadata, path_splits)
