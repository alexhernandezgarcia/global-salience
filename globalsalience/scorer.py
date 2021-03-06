"""
Pairwise Comparisons Scorer

Author: Alex Hernandez-Garcia
        alexhernandezgarcia.github.io

Last reviewed: 16 February 2020
"""

import numpy as np
import pandas as pd
from random import shuffle
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .utils import load_data
from .utils import filter_by_first_fixation, filter_by_time
from .utils import filter_by_num_fixations


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

        auc_mean : float
            Mean AUC on test

        auc_std : float
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

        self.coeff_images = None
        self.coeff_lateral = None
        self.coeff_subjects = None
        self.coeff_task = None
        self.coeff_familiarity = None

        self.zscores = None
        self.scores_scaled = None

        self.performance_metrics = None

        self.accuracy_mean = None
        self.accuracy_std = None
        self.auc_mean = None
        self.auc_std = None
        self.r2_mean = None
        self.r2_std = None
        self.mse_mean = None
        self.mse_std = None
        self.mae_mean = None
        self.mae_std = None
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
                (self.data_df.old_new_comb == 1)].index.tolist(), -1] = -1
            # - sel old (block 3) & old on left (old_new_comb 3)
            x_matrix[self.data_df.loc[
                (self.data_df.block == 3) &
                (self.data_df.old_new_comb == 3)].index.tolist(), -1] = -1
            # target on the right
            # - sel old (block 3) & old on right (old_new_comb 1)
            x_matrix[self.data_df.loc[
                (self.data_df.block == 3) &
                (self.data_df.old_new_comb == 1)].index.tolist(), -1] = 1
            # - sel new (block 2) & new on right (old_new_comb 3)
            x_matrix[self.data_df.loc[
                (self.data_df.block == 2) &
                (self.data_df.old_new_comb == 3)].index.tolist(), -1] = 1

        # Familiarity bias
        if self.familiarity_bias:
            # bias term:
            # familiarity: 1 if old on right, -1 if on left, 0 otherwise
            x_matrix = np.c_[x_matrix, np.zeros([self.n_trials, 1])]
            # old on the left (old_new_comb 3)
            x_matrix[self.data_df.loc[
                self.data_df.old_new_comb == 3].index.tolist(), -1] = -1
            # old on the right (old_new_comb 1)
            x_matrix[self.data_df.loc[
                self.data_df.old_new_comb == 1].index.tolist(), -1] = 1

        # Set images shown in each trial
        x_matrix[self.data_df.index, self.data_df.image_left - 1] = -1
        x_matrix[self.data_df.index, self.data_df.image_right - 1] = 1

        # Set subject
        if self.bias_type == 'subject':
            x_matrix[self.data_df.index, 
                     self.n_img + self.data_df.subject_index - 1] = 1

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


    @staticmethod
    def train_lin_reg(x_matrix, y):
        """
        Trains a linear regression model

        Parameters
        ----------
        x_matrix : {array-like, sparse matrix}
            Design matrix

        y: array-like
            Target labels

        Returns
        -------
        lin_reg: LinearRegression object
            LinearRegression object with the attributes of the trained model
        """

        lin_reg = LinearRegression(fit_intercept=False)
        lin_reg.fit(x_matrix, y)

        return lin_reg


    def tr_tt_split(self, test_pct):
        """
        Performs a train/test split without taking into account that data come
        from different subjects

        Parameters
        ----------
        test_pct: float [0, 1]
            Percentage of examples in the test set.

        Returns
        -------
        x_tr : {array-like, sparse matrix}
            Design matrix of the train set

        x_tt : {array-like, sparse matrix}
            Design matrix of the test set

        y_tr : array-like
            Target labels of the train set

        y_tt : array-like
            Target labels of the test set

        subj_tr : list
            List of subject indices of the train set
        """
        if self.subject_aware:
            n_subj_tr = int(np.ceil((1 - test_pct) * self.n_subj))

            rand_subj = np.random.permutation(self.subjects)

            subj_tr = rand_subj[:n_subj_tr]
            subj_tt = rand_subj[n_subj_tr:]

            idx_tr = self.data_df.loc[
                    self.data_df.subject_index.isin(subj_tr)].index
            idx_tt = self.data_df.loc[
                    self.data_df.subject_index.isin(subj_tt)].index

            x_tr = self.x_matrix[idx_tr, :]
            x_tt = self.x_matrix[idx_tt, :]
            y_tr = self.y[idx_tr]
            y_tt = self.y[idx_tt]
        else:
            x_tr, x_tt, y_tr, y_tt = train_test_split(
                    self.x_matrix, self.y, test_size=test_pct)
            subj_tr = self.subjects

        return x_tr, x_tt, y_tr, y_tt, subj_tr


    def subj_aware_cv_partitions(self, subjects):
        """
        Retrieves the train/val cross-validation partitions given a number of
        folds

        Parameters
        ----------
        subjects : list
            Indices of the subjects that are available for creating the
            partitions

        Returns
        -------
        partitions: list of tuples
            List of tuples that contain the arrays with the tr/val indices of
            the partitions
        """
        def check_partitions(partitions):
            """
            Check if the target variable contains 2 and only 2 different labels
            in both the train and the validation partitions

            Parameters
            ----------
            partitions : list of tuples
                List of tuples that contain the arrays with the tr/val indices
                of the partitions

            Returns
            -------
            bool
            """
            for p in partitions:

                idx_tr = p[0]
                idx_val = p[1]

                y_tr = self.y[idx_tr]
                y_val = self.y[idx_val]

                if (len(np.unique(y_tr)) != 2) | (len(np.unique(y_val)) != 2):
                    return False

            return True

        n_val = int(np.ceil(np.divide(float(len(subjects)), self.val_folds)))
        rand_subj = np.random.permutation(subjects)

        subj_val = []
        subj_tr = []
        for i in range(self.val_folds):
            subj_val.append(rand_subj[i * n_val:n_val + i * n_val].tolist())
            subj_tr.append(np.r_[rand_subj[:i * n_val],
                                 rand_subj[n_val + i * n_val:]].tolist())

        partitions = list(zip(subj_tr, subj_val))

        if check_partitions(partitions):
            return partitions
        else:
            return self.subj_aware_cv_partitions(subjects)


    def kfold_evaluation(self, test_pct=0.2, test_folds=10,
                         shuffle_labels=False, do_print=True):
        """
        Trains a logistic regression classifier on the eye-tracking data and
        the corresponding target variable and evaluates the performance through
        K-fold cross-evaluation, that is by creating K folds of training and
        held out test data.

        Parameters
        ----------
        test_pct : float [0, 1]
            Percentage of subjects in the test set.

        test_folds : int
            Number of random partitions of train/test sets in order the
            evaluate the performance

        do_print : bool
            If True, print a summary of the performance

        Returns
        -------
        self
        """
        def get_evaluation_metrics(estimator, x_tr, x_tt, y_tr, y_tt,
                                   dict_metrics, fold, ml_mode):
            """
            Computes a set of performance metrics from an estimator on a fold
            of train and test data.

            Parameters
            ----------
            estimator : sklearn Model
                The model to evaluate

            x_tr : array-like
                Train data

            x_tt : array-like
                Test data

            y_tr : array-like
                Train target

            y_tt : array-like
                Test target

            dict_metrics : dict
                A dictionary initialized with the keys of the performance
                metrics

            fold : int
                The index of the fold, for accessing the arrays of the
                dictionary

            ml_mode: str
                Either 'classification' or 'regression'

            Returns
            -------
            self
            """
            if ml_mode == 'classification':
                y_tr_pred_prob = estimator.predict_proba(x_tr)
                y_tt_pred_prob = estimator.predict_proba(x_tt)
            elif ml_mode == 'regression':
                y_tr_pred = estimator.predict(x_tr)
                y_tt_pred = estimator.predict(x_tt)
            else:
                raise ValueError()

            for metric_key, dict_metric in dict_metrics.items():

                # Accuracy
                if metric_key == 'accuracy': 
                    dict_metric['tr'][fold] = estimator.score(x_tr, y_tr)
                    dict_metric['tt'][fold] = estimator.score(x_tt, y_tt)

                # AUC
                elif metric_key == 'auc': 
                    dict_metric['tr'][fold] = roc_auc_score(
                            y_tr, y_tr_pred_prob[:, 1])
                    dict_metric['tt'][fold] = roc_auc_score(
                            y_tt, y_tt_pred_prob[:, 1])

                # R2 Tjur (classification)
                elif metric_key == 'r2tjur': 
                    dict_metric['tr'][fold] = \
                            np.mean(y_tr_pred_prob[y_tr == 1], axis=0)[1] - \
                            np.mean(y_tr_pred_prob[y_tr == 0], axis=0)[1]
                    dict_metric['tt'][fold] = \
                            np.mean(y_tt_pred_prob[y_tt == 1], axis=0)[1] - \
                            np.mean(y_tt_pred_prob[y_tt == 0], axis=0)[1]

                # Log Loss
                elif metric_key == 'logloss': 
                    dict_metric['tr'][fold] = log_loss(y_tr, y_tr_pred_prob)
                    dict_metric['tt'][fold] = log_loss(y_tt, y_tt_pred_prob)

                # Bayesian information criterion
                elif metric_key == 'bic': 
                    dict_metric['tr'][fold] = \
                            -2 * np.sum(np.log(y_tr_pred_prob[:, 1])) + \
                            x_tr.shape[1] * np.log(x_tr.shape[0])
                    dict_metric['tt'][fold] = \
                            -2 * np.sum(np.log(y_tt_pred_prob[:, 1])) + \
                            x_tt.shape[1] * np.log(x_tt.shape[0])

                # Akaike information criterion
                elif metric_key == 'aic': 
                    dict_metric['tr'][fold] = \
                            -2 * np.sum(np.log(y_tr_pred_prob[:, 1])) + \
                            2 * x_tr.shape[1]
                    dict_metric['tt'][fold] =  \
                            -2 * np.sum(np.log(y_tt_pred_prob[:, 1])) + \
                            2 * x_tt.shape[1]

                # R2 coefficient (regression)
                elif metric_key == 'r2': 
                    dict_metric['tr'][fold] = estimator.score(x_tr, y_tr)
                    dict_metric['tt'][fold] = estimator.score(x_tt, y_tt)

                # Mean squared error
                elif metric_key == 'mse': 
                    dict_metric['tr'][fold] = mean_squared_error(
                            y_tr, y_tr_pred)
                    dict_metric['tt'][fold] = mean_squared_error(
                            y_tt, y_tt_pred)
                    
                # Mean absolute error
                elif metric_key == 'mae': 
                    dict_metric['tr'][fold] = mean_absolute_error(
                            y_tr, y_tr_pred)
                    dict_metric['tt'][fold] = mean_absolute_error(
                            y_tt, y_tt_pred)

                else: 
                    raise NotImplementedError()

            return dict_metrics

        def print_performance_metrics(dict_metrics, ml_mode):
            metrics_print = {'accuracy': 'accuracy',
                             'auc': 'AUC',
                             'r2tjur': 'R2 (Tjur)',
                             'logloss': 'log loss',
                             'bic': 'BIC',
                             'aic': 'AIC',
                             'r2': 'R2',
                             'mse': 'MSE',
                             'mae': 'MAE'}

            for metric_key, dict_metric in dict_metrics.items():
                print('Training {}: {:.4f} (std = {:.4f})'.format(
                    metrics_print[metric_key],
                    np.mean(dict_metric['tr']),
                    np.std(dict_metric['tr'])))
                print('Test {}: {:.4f} (std = {:.4f})'.format(
                    metrics_print[metric_key],
                    np.mean(dict_metric['tt']),
                    np.std(dict_metric['tt'])))
                print('')

        # Check target
        if self.target in ['first', 'longer', 'longer_avg_dur', 'more']:
            ml_mode = 'classification'
            metrics = ['accuracy', 'auc', 'r2tjur', 'logloss', 'bic', 'aic']
        elif self.target in ['time', 'number']:
            ml_mode = 'regression'
            metrics = ['r2', 'mse', 'mae']
        else:
            raise NotImplementedError(
                'Logistic regression classification can only be '
                'trained with binary targets: first, longer, '
                'longer_avg_dur, more; Linear regression can only be '
                'trained with scalar targets: time, number ')

        self.performance_metrics = {k: {'mean': None, 'std': None} 
                for k in metrics}
        dict_metrics = {k: {'tr': np.zeros(test_folds), 
                            'tt': np.zeros(test_folds)} for k in metrics}

        for fold in tqdm(range(test_folds)):

            x_tr, x_tt, y_tr, y_tt, subj_tr = self.tr_tt_split(test_pct)

            # Compute random baseline
            if shuffle_labels:
                shuffle(y_tr.values)

	    # Train model
            if self.subject_aware and not shuffle_labels:
                if ml_mode == 'regression':
                    estimator = self.train_lin_reg(self.x_matrix, self.y)
                elif ml_mode == 'classification':
                    cv = self.subj_aware_cv_partitions(subj_tr)
                    gscv = self.train_cv_log_reg(self.x_matrix, self.y, cv)
                    estimator = gscv.best_estimator_
                else:
                    raise ValueError()
            else:
                if ml_mode == 'regression':
                    estimator = self.train_lin_reg(x_tr, y_tr)
                elif ml_mode == 'classification':
                    cv = self.val_folds
                    gscv = self.train_cv_log_reg(x_tr, y_tr, cv)
                    estimator = gscv.best_estimator_
                else:
                    raise ValueError()

            # Compute performance metrics and update dictionary
            dict_metrics = get_evaluation_metrics(estimator, x_tr, x_tt, y_tr,
                                                  y_tt, dict_metrics, fold,
                                                  ml_mode)

        # Compute mean and standard deviation of metrics across the folds
        for k, v in dict_metrics.items():
            self.performance_metrics[k]['mean'] = np.mean(v['tt'])
            self.performance_metrics[k]['std'] = np.std(v['tt'])

        if do_print:
            print_performance_metrics(dict_metrics, ml_mode)

        return self


    def compute_scores(self):
        """
        Trains the model on all the available training data, obtains the
        coefficients of the model and normalizes them to obtain useful scores

        Returns
        -------
        self
        """

        # Define cross-validation partitions
        if self.subject_aware:
            cv = self.subj_aware_cv_partitions(self.subjects)
        else:
            cv = self.val_folds

        # Train logistic regression
        gscv = self.train_cv_log_reg(self.x_matrix, self.y, cv)
        estimator = gscv.best_estimator_

        # Get coefficients of the regression model
        coefficients = estimator.coef_.squeeze()
        self.coeff_images = coefficients[:self.n_img]

        # Normalise
        zscores = self.coeff_images.copy()
        zscores -= zscores.mean()
        self.zscores = zscores / zscores.std()

        # Scale
        scores_scaled = self.coeff_images.copy()
        scores_scaled -= scores_scaled.min()
        scale = scores_scaled.max() - scores_scaled.min()
        self.scores_scaled = scores_scaled / scale

        # Lateral (subject) bias
        if self.bias_type == 'none':
            pass
        elif self.bias_type == 'global':
            self.coeff_lateral = coefficients[self.n_img]
        elif self.bias_type == 'subject':
            self.coeff_subjects = \
                coefficients[self.n_img:self.n_img + self.n_subj]
        else:
            raise NotImplementedError()

        if self.familiarity_bias:
            self.coeff_familiarity = coefficients[-1]
            if self.task_bias:
                self.coeff_task = coefficients[-2]
        else:
            if self.task_bias:
                self.coeff_task = coefficients[-1]

