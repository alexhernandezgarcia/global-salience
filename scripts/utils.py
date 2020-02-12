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
        data_df = pd.read_csv(filename)
    else:
        raise IOError('The filename must have a .mat or .p extension')

    if get_all_data:
        data_df = get_all_keys(data_df, categories_yml)

    # Set correct dtypes
    data_df.astype({'fix': 'int32',
                    'start': 'int32',
                    'end': 'int32',
                    'start_lefteye': 'int32',
                    'end_lefteye': 'int32',
                    'start_righteye': 'int32',
                    'end_righteye': 'int32',
                    'duration': 'int32'})

    if output_csv:
        data_df.to_csv(output_csv)

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
    return data_df.dropna(axis=0, subset=['x', 'y']).copy()


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
                (((data_df.x >= 0) & (data_df.x <= max_x_left)) | 
                 ((data_df.x >= min_x_right) & (data_df.x <= screen_width))) &
                (data_df.duration >= 50) & 
                (data_df.duration < duration_mean + 2 * duration_std), 
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
                (data_df.right_is_new == False), 'old_new_comb'] = 2
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
