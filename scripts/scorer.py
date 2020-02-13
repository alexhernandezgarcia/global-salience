"""
Pairwise Comparisons Scorer

Author: Alex Hernandez-Garcia
        alexhernandezgarcia.github.io

Last reviewed: 13 February 2020
"""

import numpy as np
import pandas as pd

from utils import load_data
from utils import filter_by_first_fixation, filter_by_time
from utils import filter_by_num_fixations

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


class PairwiseComparisonsScorer:
    """
    Pairwise Comparisons (a.k.a. Paired Comparisons or 2 Alternative Forced
    Choice - 2AFC) Scorer.

    In pairwise comparisons experiments elements are judged pairwise, normally
    by subjects who consciously or unconsciously decide in favor of one of the
    two elements of the pair.

    This class implements the functionality to derive global scores for each of
    the images used in an eye-tracking experiment where participants were shown
    pairs of images side-by-side.

    Read more in this tutorial:
    `How to get global scores from pairwise comparison experiments data
    doc/tutorial_paired_comparisons.ipynb>`.

    References
    ----------
    Tutorial -- How to get global scores from pairwise comparison experiments
                data
        /doc/tutorial_paired_comparisons.ipynb

    About the coefficient of determination Tjur
        http://statisticalhorizons.com/r2logistic
    """

    def __init__(self, data=None, bias_type='subject', subject_aware=True,
                 val_folds=5, target='first', valid=True, task_bias=True,
                 familiarity_bias=True):
        """
        Parameters
        ----------
        data : str
            Path to a .mat or .csv file containing the data set of fixations, or
            None.

        bias_type : str
            One of: None or 'global' or 'subject'
            Specifies the type of bias terms that should be added to the model.
            Read the tutorial for more information.

        subject_aware : bool
            Specifies whether the cross-validation/evaluation partitions must
            be aware of the data coming from different subjects and prevents
            train and test sets from containing data from same subjects.

        val_folds : int
            Number of folds in the parameter cross-validation partitions

        target : str
            One of: 'first' or 'longer' or 'more' or 'time' or 'number'
            Specifies the target variable for the analysis of information that
            must be filter and retrieved. Options are:

                'first' : side (left/right) of the first fixation (binary)

                'longer' : side fixated longer time (binary)

                'more' : side with higher number of fixations (binary)

                'longer_avg_dur' : side with a longer average fixation duration
                (binary)

                'time' : percentage of time on the right image (scalar)

                'number' : percentage of fixations on the right image (scalar)

                'avg_dur' : average fixation duration (scalar)

        valid : bool
            Whether only the trials marked as 'valid' must be used.

        task_bias : bool
            Whether to use a bias term to model the task of the participants at
            each trial.

        familiarity_bias : bool
            Whether to use a bias term to model the familiarity effect.

        Attributes
        ----------
        weights : ndarray, shape (n_img,)
            Coefficients vector

        bias : ndarray, shape (n_subj,) or (1,) or []
            Bias terms, depending on bias_type, it will be of different shape

        scores_zscores : ndarray, shape (n_img,)
            Coefficients vector, normalized as standard scores

        scores_scaled : ndarray, shape (n_img,)
            Coefficients vector, normalized to range in [0, 1]

        accuracy_mean : float
            Mean accuracy on test

        accuracy_std : float
            Standard deviation of the accuracy on test

        AUC_mean : float
            Mean AUC on test

        AUC_std : float
            Standard deviation of the AUC on test

        r2_mean : float
            Mean R2 on test

        r2_std : float
            Standard deviation of the R2 on test

        log_loss_mean : float
            Mean logarithmic loss on test

        log_loss_std : float
            Standard deviation of the logarithmic loss on test
        """
        self.bias_type = bias_type
        self.subject_aware = subject_aware
        self.val_folds = val_folds
        self.target = target
        self.valid = valid
        self.task_bias = task_bias
        self.familiarity_bias = familiarity_bias

        if target == 'first':
            self.filter_type = 'first_fix'
        elif target == 'time' or target == 'longer' or \
             target == 'longer_avg_dur' or target == 'avg_dur':
            self.filter_type = 'time'
        elif target == 'number' or target == 'more':
            self.filter_type = 'num_fixations'
        else:
            raise ValueError()

        self.data_df = None
        self.x_matrix = None
        self.y = None

        self.n_subj = None
        self.n_img = None
        self.n_trials = None
        self.subjects = None

        self.weights = None
        self.bias = None
        self.scores_zscores = None
        self.scores_scaled = None
        self.accuracy_mean = None
        self.accuracy_std = None
        self.AUC_mean = None
        self.AUC_std = None
        self.r2_mean = None
        self.r2_std = None
        self.log_loss_mean = None
        self.log_loss_std = None
        self.coef_det_mean = None
        self.coef_det_std = None

        self.bic_mean = None
        self.bic_std = None
        self.aic_mean = None
        self.aic_std = None

        if data is not None:
            self.load_data(data)


    def load_data(self, filename, compute_features=True, do_filter=True):
        """
        Reads the file that contains the experimental data, filters them
        and initializes the necessary variables

        Parameters
        ----------
        filename : str
            Path to the file containing the experimental data

        Returns
        -------
        self
        """
        self.data_df = load_data(filename, get_all_data=compute_features)

        if do_filter:
            self.data_df = self.filter_data(self.data_df)

        self.subjects = self.data_df.subject_index.unique()
        self.n_subj = len(self.subjects)
        self.n_img = len(self.data_df.image_left.unique())
        self.n_trials = len(self.data_df.idx_trial_subj)

        self.x_matrix, self.y = self.compute_design_matrix()

        return self


    def filter_data(self, data_df):
        """
        Filters the data set to keep only the relevant information for the
        desired purposes

        Parameters
        ----------
        data_df : DataFrame
            DataFrame containing the data set

        Returns
        -------
        data_df_filtered : DataFrame
            Filtered DataFrame containing the useful experimental data.
        """
        if self.filter_type == 'first_fix':
            data_df_filtered = filter_by_first_fixation(data_df, self.valid)
        elif self.filter_type == 'time':
            data_df_filtered = filter_by_time(data_df)
        elif self.filter_type == 'num_fixations':
            data_df_filtered = filter_by_num_fixations(data_df)
        else:
            raise NotImplementedError('filter_type can be first_fix, time or '\
                                      'num_fixations')
        
        # Reset index
        data_df_filtered.reset_index(drop=True, inplace=True)

        return data_df_filtered


    def compute_design_matrix(self):
        """
        Computes the design matrix

        Note that the order of the coefficients is:
        | images | lateral (subject) | task | familiarity |
        which is different to the order illustrated in the manuscript

        Parameters
        ----------

        Returns
        -------
        x_matrix : {array-like, sparse matrix}
            Design matrix

        y: array-like
            Target labels
        """
        # Lateral (subject) bias
        if self.bias_type == 'none':
            x_matrix = np.zeros([self.n_trials, self.n_img])
        elif self.bias_type == 'global':
            x_matrix = np.c_[np.zeros([self.n_trials, self.n_img]), 
                             np.ones([self.n_trials, 1])]
        elif self.bias_type == 'subject':
            x_matrix = np.c_[np.zeros([self.n_trials, self.n_img]), 
                             np.zeros([self.n_trials, self.n_subj])]
        else:
            raise NotImplementedError()

        # Task bias 
        if self.task_bias:
            # bias term:
            # target: 1 if target right, -1 if target left, 0 otherwise
            x_matrix = np.c_[x_matrix, np.zeros([self.n_trials, 1])]

            # target on the left
            # - sel new (block 2) & new on left (old_new_comb 1)
            x_matrix[self.data_df.loc[
                (self.data_df.block == 2) & 
                (self.data_df.old_new_comb == 1)].index.tolist()] = -1 
            # - sel old (block 3) & old on left (old_new_comb 3)
            x_matrix[self.data_df.loc[
                (self.data_df.block == 3) & 
                (self.data_df.old_new_comb == 3)].index.tolist()] = -1 
            # target on the right
            # - sel old (block 3) & old on right (old_new_comb 1)
            x_matrix[self.data_df.loc[
                (self.data_df.block == 3) & 
                (self.data_df.old_new_comb == 1)].index.tolist()] = 1 
            # - sel new (block 2) & new on right (old_new_comb 3)
            x_matrix[self.data_df.loc[
                (self.data_df.block == 2) & 
                (self.data_df.old_new_comb == 3)].index.tolist()] = 1 

        # Familiarity bias
        if self.familiarity_bias:
            # bias term:
            # familiarity: 1 if old on right, -1 if on left, 0 otherwise
            x_matrix = np.c_[x_matrix, np.zeros([self.n_trials, 1])]
            # old on the left (old_new_comb 3)
            x_matrix[self.data_df.loc[
                self.data_df.old_new_comb == 3].index.tolist()] = -1
            # old on the right (old_new_comb 1)
            x_matrix[self.data_df.loc[
                self.data_df.old_new_comb == 1].index.tolist()] = 1

        # Set images shown in each trial
        for t in range(0, self.n_trials):
            x_matrix[t, self.data_df.iloc[t]['image_left'] - 1] = -1
            x_matrix[t, self.data_df.iloc[t]['image_right'] - 1] = 1

            if self.bias_type == 'subject':
                x_matrix[t, self.n_img + np.where(self.subjects == \
                        self.data_df.iloc[t]['subject_index'])[0]] = 1

        # Target (y)
        if self.target == 'first':
            y = self.data_df.is_right
        elif self.target == 'longer':
            y = self.data_df.longer_right
        elif self.target == 'longer_avg_dur':
            y = self.data_df.longer_avg_dur_right
        elif self.target == 'more':
            y = self.data_df.more_right
        elif self.target == 'time':
            y = self.data_df.time_right
        elif self.target == 'number':
            y = self.data_df.n_fixations_right
        elif self.target == 'avg_dur':
            y = self.data_df.avg_dur_right
        else:
            raise ValueError('Please specify a valid target type')

        return x_matrix, y


    @staticmethod
    def train_cv_log_reg(x_matrix, y, cv):
        """
        Trains a logistic regression classifier by cross-validating the C
        parameter through grid search

        Parameters
        ----------
        x_matrix : {array-like, sparse matrix}
            Design matrix

        y: array-like
            Target labels

        cv: int or an iterable
            Determines the cross-validation strategy. If integer, it will be
            the number of folds; if iterable, it will be some previously
            defined partitions.

        Returns
        -------
        gscv: GridSearchCV object
            GridSearchCV object that contains all the details of the training
            and cross-validation process

        See
        ---
        scikit-learn.org

            LogisticRegression : https://scikit-learn.org/stable/modules/ ...
                generated/sklearn.linear_model.LogisticRegression.html

            GridSearchCV : https://scikit-learn.org/stable/modules/ ...
                generated/sklearn.model_selection.GridSearchCV.html
        """
        # Range of C
        min_c_pow = -3
        max_c_pow = 3
        num_c = 10
        c_range = np.logspace(min_c_pow, max_c_pow, num=num_c)
        parameters = {'C': c_range}

        log_reg = LogisticRegression(fit_intercept=False, max_iter=500)
        gscv = GridSearchCV(log_reg, parameters, cv=cv, scoring='roc_auc')
        gscv.fit(x_matrix, y)

        return gscv

