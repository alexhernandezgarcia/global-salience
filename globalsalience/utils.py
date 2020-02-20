"""
Utils for the PairwiseComparisonsScorer

Author: Alex Hernandez-Garcia
        alexhernandezgarcia.github.io

Last reviewed: 18 February 2020
"""

import scipy.io as sio
import numpy as np
import pandas as pd
import itertools
import os.path
import yaml
from tqdm import tqdm


def load_data(filename, mat_variable_name='', output_csv=None,
              get_all_data=True, categories_yml='../data/categories.yml'):
    """
    Loads a data file, either MATLAB .mat or .csv, and returns a pandas
    DataFrame

    Parameters
    ----------
    filename : str
        Path to the data file

    mat_variable_name : str
        Name of the variable to be retrieved as data from the .mat file. If
        not specified, the first variable of the file is used.

    output_csv : str
        File name of the output csv. None if not save.

    get_all_data : bool
        Specifies if the additional keys must be incorporated to the dictionary.

    categories_yml : str
        Path to the YAML file that specifies the category of each image

    Returns
    -------
    data_df : dict
        DataFrame containing the eye-tracking data
    """
    if not os.path.isfile(filename):
        raise ValueError('Please provide an existing data_file')

    if filename[-4:] == '.mat':
        data_df = mat2df(filename, mat_variable_name)
    elif filename[-4:] == '.csv':
        data_df = pd.read_csv(filename, index_col='index')
    else:
        raise IOError('The filename must have a .mat or .p extension')

    if get_all_data:
        data_df = get_all_keys(data_df, categories_yml)

    # Set correct dtypes
    dtypes_dict = {'fix': 'int32',
                   'start': 'int32',
                   'end': 'int32',
                   'start_lefteye': 'int32',
                   'end_lefteye': 'int32',
                   'start_righteye': 'int32',
                   'end_righteye': 'int32',
                   'duration': 'int32'}
    for k, v in dtypes_dict.items():
        if k in data_df:
            data_df.astype({k: v})

    if output_csv:
        data_df.to_csv(output_csv, index_label='index')

    return data_df


def mat2df(input_mat, variable_name=''):
    """
    Converts a MATLAB .mat fixations file into a pandas DataFrame

    Parameters
    ----------
    input_mat : str
        Path to the MATLAB .mat file

    variable_name : str
        Name of the variable to be retrieved as data from the .mat file. If
        not specified, the first variable of the file is used.

    Returns
    -------
    output_dict : DataFrame
        Pandas DataFrame built from the .mat file
    """

    assert input_mat[-4:] == '.mat'

    # Read mat file
    mat_file = sio.loadmat(input_mat)

    # Set variable name to be retrieved from the mat file
    if variable_name == '':
        variable_name = [k for k in mat_file.keys() if k[:2] != '__'][0]

    # Parse data into a dictionary
    fixations_data = mat_file[variable_name]
    data_keys = fixations_data.dtype.names
    data_dict = dict.fromkeys(data_keys)
    for key in data_keys:
        data_dict[key] = fixations_data[0][key][0][0]

    # Substitute 'subjectindex' key by 'subject_index'
    if 'subjectindex' in data_dict.keys():
        data_dict['subject_index'] = data_dict.pop('subjectindex')

    data_df = pd.DataFrame.from_dict(data_dict)

    return data_df


def get_all_keys(data_df, categories_yml=None):
    """
    Compute all the implemented features and add them to the DataFrame as new
    columns.

    Parameters
    ----------
    data_df : DataFrame
        pandas DataFrame

    categories_file : str
        File containing the mappings between images and categories

    Returns
    -------
    data_df : DataFrame
        Updated DataFrame
    """
    data_df = delete_nan(data_df)
    data_df = get_valid(data_df)
    data_df = get_fixation_side(data_df)
    data_df = get_idx_trial_subj(data_df)
    data_df = get_is_new(data_df)
    data_df = get_categories(data_df, categories_yml)

    return data_df


def delete_nan(data_df):
    """
    Delete fixations with fixation locations x or y as nan.

    Parameters
    ----------
    data_df : DataFrame
        pandas DataFrame

    Returns
    -------
    data_df : DataFrame
        Updated DataFrame
    """
    data_df = data_df.dropna(axis=0, subset=['x', 'y']).copy()
    data_df.reset_index(drop=True, inplace=True)

    return data_df



def get_valid(data_df, screen_width=3840, screen_height=2160,
              gap_pct=160.0/2560.0, force=False):
    """
    Add a new column to indicate the validity of the fixations. Valid fixations
    are those within the limits of the experimental screen, beyond the central
    gap between the two images and with a duration longer than 50 ms and
    within 2 standard deviations of the average fixation duration.

    Parameters
    ----------
    data_df : DataFrame
        pandas DataFrame

    screen_width : int
        Width of the experimental display in pixels

    screen_height : int
        Height of the experimental display in pixels

    gap_pct : float
        Percentage of the display width between the left and right images

    force : bool
        Whether forcing the computation of validity (relevant because of the
        calculation of duration statistics)

    Returns
    -------
    data_df : DataFrame
        Updated DataFrame
    """
    if 'valid' in data_df and not force:
        return data_df

    if 'duration' not in data_df:
        data_df = get_duration(data_df)
    duration_mean = data_df.duration.mean()
    duration_std = data_df.duration.std()

    # Gap coordinates
    gap = round(gap_pct * screen_width)
    max_x_left = np.divide((screen_width - gap), 2)
    min_x_right = max_x_left + gap

    # Conditions for validity:
    #   - Fixation location must be within the limits of the screen
    #   - Fixation location must not be beyond the gap between the images
    #   - Fixation duration must be longer than 50 ms
    #   - Fixation duration must be within 2 sigmas of the mean duration
    data_df.loc[:, 'valid'] = False
    data_df.loc[(data_df.y >= 0) & (data_df.y <= screen_height) &
                (data_df.x >= 0) & (data_df.x <= screen_width) &
                ((data_df.x <= max_x_left) | (data_df.x >= min_x_right)) & 
                (data_df.duration >= 50) &
                (data_df.duration <= duration_mean + 2 * duration_std),
                'valid'] = True

    return data_df


def get_fixation_side(data_df, screen_width=3840):
    """
    Add new columns to indicate the side (left or right image) of the fixation.
	is_left : True if the fixation is on the left image, False otherwise
	is_right : True if the fixation is on the right image, False otherwise

    Parameters
    ----------
    data_df : DataFrame
        pandas DataFrame

    screen_width : int
        Width of the experimental display in pixels

    Returns
    -------
    data_df : DataFrame
        Updated DataFrame
    """
    data_df.loc[:, 'is_left'] = False
    data_df.loc[data_df.x <= screen_width / 2, 'is_left'] = True
    data_df.loc[:, 'is_right'] = False
    data_df.loc[data_df.x > screen_width / 2, 'is_right'] = True

    return data_df


def get_idx_trial_subj(data_df):
    """
    Add a new column that assigns every data point (fixation) of the same pair
    of images and subject a unique identifier. New column is:
        idx_trial_subj : unique identifier of every pair of images and subject

    Parameters
    ----------
    data_df : DataFrame
        pandas DataFrame

    Returns
    -------
    data_df : DataFrame
        Updated DataFrame
    """
    data_df.loc[:, 'idx_trial_subj'] = None
    for idx, (trial, subj) in enumerate(
		tqdm(itertools.product(data_df.trial.unique(),
                                       data_df.subject_index.unique()))):
        data_df.loc[(data_df.trial == trial) &
                    (data_df.subject_index == subj), 'idx_trial_subj'] = idx

    return data_df


def get_duration(data_df):
    """
    Add a column with the duration of the fixations.

    Parameters
    ----------
    data_df : DataFrame
        pandas DataFrame

    Returns
    -------
    data_df : DataFrame
        Updated DataFrame
    """
    if 'duration' in data_df:
        return data_df

    data_df.loc[:, 'duration'] = data_df.end - data_df.start

    return data_df


def get_is_new(data_df):
    """
    Add new columns to indicate whether the images are new (not yet seen by the
    participant) or old (already seen) as well as the type. The new columns
    are:
        left_is_new: True if the left image is new
        left_is_old: True if the left image is old
        old_new_comb: type of left-right combination regarding familiarity:
            1: new old
            2: old old
            3: old new
            4: new new

    Parameters
    ----------
    data_df : DataFrame
        pandas DataFrame

    Returns
    -------
    data_df : DataFrame
        Updated DataFrame
    """

    data_df.loc[:, 'left_is_new'] = False
    data_df.loc[:, 'right_is_new'] = False

    for subj in tqdm(data_df.subject_index.unique()):

        id_trial_subj = data_df.loc[
                (data_df.subject_index == subj)]['idx_trial_subj'].unique()

        images_seen = []

        for idx in id_trial_subj:
            if data_df.loc[
                    data_df.idx_trial_subj == idx]['image_left'].iloc[0] \
                            not in images_seen:
                data_df.loc[data_df.idx_trial_subj == idx,
                            'left_is_new'] = True
                images_seen.append(data_df.loc[
                    data_df.idx_trial_subj == idx]['image_left'].iloc[0])

            if data_df.loc[
                    data_df.idx_trial_subj == idx]['image_right'].iloc[0] \
                            not in images_seen:
                data_df.loc[data_df.idx_trial_subj == idx,
                            'right_is_new'] = True
                images_seen.append(data_df.loc[
                    data_df.idx_trial_subj == idx]['image_right'].iloc[0])

    # Add column with old/new combination to facilitate indexing
    # - 1: new old
    # - 2: old old
    # - 3: old new
    # - 4: new new
    data_df.loc[:, 'old_new_comb'] = 0
    data_df.loc[(data_df.left_is_new == True) &
                (data_df.right_is_new == False), 'old_new_comb'] = 1
    data_df.loc[(data_df.left_is_new == False) &
                (data_df.right_is_new == False), 'old_new_comb'] = 2
    data_df.loc[(data_df.left_is_new == False) &
                (data_df.right_is_new == True), 'old_new_comb'] = 3
    data_df.loc[(data_df.left_is_new == True) &
                (data_df.right_is_new == True), 'old_new_comb'] = 4

    return data_df


def get_categories(data_df, categories_yml):
    """
    Add new columns to indicate the image category of the left and right
    images.

    Parameters
    ----------
    data_df : DataFrame
        pandas DataFrame

    categories_yml: str
        YAML file containing a mapping of the categories and image indices

    Returns
    -------
    data_df : DataFrame
        Updated DataFrame
    """

    with open(categories_yml, 'r') as f:
        categories_dict = yaml.load(f, Loader=yaml.FullLoader)

        data_df.loc[:, 'category_left'] = None
        data_df.loc[:, 'category_right'] = None

        for k, v in categories_dict.items():
            if k != 'ambiguous':
                data_df.loc[data_df.image_left.isin(v), 'category_left'] = k
                data_df.loc[data_df.image_right.isin(v), 'category_right'] = k

    return data_df


def filter_by_first_fixation(data_df, valid=True):
    """
    Keep only the data points corresponding to the first fixations of each pair
    of images and subjects. The attentional engagement is also computed.

    Parameters
    ----------
    data_df : DataFrame
        pandas DataFrame containing the dataset

    valid : bool
        Specifies whether only the trials marked as 'valid' must be used.

    Returns
    -------
    data_df : DataFrame
        Updated DataFrame
    """
    # Compute first the attentional engagement (time until fixating away)
    engagement = []
    for idx in tqdm(data_df.idx_trial_subj.unique()):
        df_idx = data_df.loc[
                (data_df.idx_trial_subj == idx) & (data_df.fix >= 2),
                ['is_left', 'duration', 'valid', 'fix']]

        if valid and len(df_idx.loc[(df_idx.fix == 2) &
                                    (df_idx.valid == True)]) == 0:
            continue

        n_fix_on_first = 0
        for is_left in df_idx.is_left:
            if is_left == df_idx.is_left.iloc[0]:
                n_fix_on_first += 1

        # If not all the fixations of the sequence are valid, then the
        # engagement of the trial is not computed (-1)
        df_engagement = df_idx.iloc[:n_fix_on_first, :]
        if valid and df_engagement.valid.all():
            engagement.append(df_engagement.duration.sum())
        else: engagement.append(-1)

    # Keep only first fixation
    if valid:
        data_df = data_df.loc[(data_df.fix == 2) & (data_df.valid == True), :]
    else:
        data_df = data_df.loc[(data_df.fix == 2), :]
    data_df = data_df.drop(['fix'], axis=1)

    # Add engagement data
    data_df.loc[:, 'engagement'] = engagement

    # Remove irrelevant columns
    data_df = data_df.drop(['start_lefteye', 'end_lefteye', 'x_lefteye',
        'y_lefteye', 'pupil_lefteye', 'start_righteye', 'end_righteye',
        'x_righteye', 'y_righteye', 'pupil_righteye'], axis=1)
    return data_df.copy()


def filter_by_time(data_df):
    """
    Keep only one data point per pair of images and subject, retrieving new
    information regarding the time every side (left/right) is fixated. New keys
    are:

        time_left : time that the left image is fixated divided by the total
        duration of the pair

        time_right : time that the right image is fixated divided by the total
        duration of the pair

        longer_left : 1 if the left image was fixated longer, 0 otherwise

        longer_right : 1 if the right image was fixated longer, 0 otherwise

    Other possibly relevant and meaningful keys are included, whereas those
    specific to particular fixations are not included anymore.

    Parameters
    ----------
    data_df : DataFrame
        pandas DataFrame containing the dataset

    Returns
    -------
    data_df : DataFrame
        Updated DataFrame
    """
    for idx in tqdm(data_df.idx_trial_subj.unique()):
        df_idx = data_df.loc[
                (data_df.idx_trial_subj == idx) & (data_df.fix >= 2),
                ['is_left', 'is_right', 'duration', 'fix']]

        time_total = float(df_idx.duration.sum())
        time_left = np.divide(
                df_idx.loc[df_idx.is_left == True]['duration'].sum(),
                time_total)
        time_right = np.divide(
                df_idx.loc[df_idx.is_right == True]['duration'].sum(),
                time_total)
        avg_dur_left = df_idx.loc[df_idx.is_left == True]['duration'].mean()

        avg_dur_right = df_idx.loc[df_idx.is_right == True]['duration'].mean()
        data_df.loc[data_df.idx_trial_subj == idx, 'time_left'] = time_left
        data_df.loc[data_df.idx_trial_subj == idx, 'time_right'] = time_right
        data_df.loc[data_df.idx_trial_subj == idx,
                    'longer_left'] = time_left > time_right
        data_df.loc[data_df.idx_trial_subj == idx,
                    'longer_right'] =  time_right > time_left
        data_df.loc[data_df.idx_trial_subj == idx,
                'avg_dur_left'] = avg_dur_left
        data_df.loc[data_df.idx_trial_subj == idx,
                'avg_dur_right'] = avg_dur_right
        data_df.loc[data_df.idx_trial_subj == idx,
                    'longer_avg_dur_left'] = avg_dur_left > avg_dur_right
        data_df.loc[data_df.idx_trial_subj == idx,
                    'longer_avg_dur_right'] =  avg_dur_right > avg_dur_left

    # Keep only one data point per trial/subj and remove irrelevant columns
    data_df = data_df.loc[data_df.fix == 2, :]
    data_df = data_df.drop(['duration', 'start_lefteye', 'end_lefteye',
        'x_lefteye', 'y_lefteye', 'pupil_lefteye', 'start_righteye',
        'end_righteye', 'x_righteye', 'y_righteye', 'pupil_righteye', 'start',
        'end', 'x', 'y', 'pupil', 'fix', 'is_left', 'is_right'], axis=1)

    return data_df.copy()


def filter_by_num_fixations(data_df):
    """
    Keep only one data point per pair of images and subject, retrieving new
    information regarding the number of fixations every side (left/right)
    received. New keys are:

        pct_left : number of fixations that the left image received
        divided by the total number of fixations of the pair

        pct_right : number of fixations that the right image received
        divided by the total number of fixations of the pair

        more_left : 1 if the left image was fixated more times, 0 otherwise

        more_right : 1 if the right image was fixated more times, 0 otherwise

    Other possibly relevant and meaningful keys are included, whereas those
    specific to particular fixations are not included anymore.

    Parameters
    ----------
    data_df : DataFrame
        pandas DataFrame containing the dataset

    Returns
    -------
    data_df : DataFrame
        Updated DataFrame
    """
    for idx in tqdm(data_df.idx_trial_subj.unique()):
        df_idx = data_df.loc[
                (data_df.idx_trial_subj == idx) & (data_df.fix >= 2),
                ['is_left', 'is_right', 'fix']]

        n_fix_total = len(df_idx)
        pct_left = np.divide(df_idx.is_left.sum(), n_fix_total)
        pct_right = np.divide(df_idx.is_right.sum(), n_fix_total)

        data_df.loc[data_df.idx_trial_subj == idx, 'pct_left'] = pct_left
        data_df.loc[data_df.idx_trial_subj == idx, 'pct_right'] = pct_right
        data_df.loc[data_df.idx_trial_subj == idx,
                    'more_left'] = pct_left > pct_right
        data_df.loc[data_df.idx_trial_subj == idx,
                    'more_right'] = pct_right > pct_left

    # Keep only one data point per trial/subj and remove irrelevant columns
    data_df = data_df.loc[data_df.fix == 2, :]
    data_df = data_df.drop(['duration', 'start_lefteye', 'end_lefteye',
        'x_lefteye', 'y_lefteye', 'pupil_lefteye', 'start_righteye',
        'end_righteye', 'x_righteye', 'y_righteye', 'pupil_righteye', 'start',
        'end', 'x', 'y', 'pupil', 'fix', 'is_left', 'is_right'], axis=1)

    return data_df.copy()

