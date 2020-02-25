import argparse

import numpy as np
import pandas as pd

from globalsalience.scorer import PairwiseComparisonsScorer

# Initialize the Flags container
FLAGS = None

def main(argv=None):

    # Initialize the scorer and load the first fixations data set
    scorer = PairwiseComparisonsScorer(target='first')
    scorer.load_data(FLAGS.input, compute_features=False, do_filter=False)

    # Initialize DataFrame
    columns_zscores = ['zscores_{:d}'.format(idx + 1) for idx 
            in range(scorer.n_img)]
    if scorer.bias_type == 'none':
        columns_lateral = []
    elif scorer.bias_type == 'global':
        columns_lateral = ['global']
    elif scorer.bias_type == 'subject':
        columns_lateral = ['subject{:d}'.format(idx + 1) for idx 
            in range(scorer.n_subj)]
    else:
        raise NotImplementedError()
    columns = columns_zscores + columns_lateral + ['task', 'familiarity']
    df_scores = pd.DataFrame()

    # Compute scores
    for rep in range(FLAGS.n_rep):
        scorer.compute_scores()
        if scorer.bias_type == 'none':
            mat = np.hstack((scorer.zscores, scorer.coeff_task, 
                             scorer.coeff_familiarity))
        elif scorer.bias_type == 'global':
            mat = np.hstack((scorer.zscores, scorer.coeff_lateral, 
                             scorer.coeff_task, scorer.coeff_familiarity))
        elif scorer.bias_type == 'subject':
            mat = np.hstack((scorer.zscores, scorer.coeff_subjects, 
                             scorer.coeff_task, scorer.coeff_familiarity))
        else:
            raise NotImplementedError()
        d = pd.DataFrame(data=np.expand_dims(mat, axis=0),
                         columns=columns,
                         index=[rep])
        df_scores = df_scores.append(d)

    # Store CSV
    df_scores.to_csv(FLAGS.output, index_label='index')
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        default='../data/data_firstfixation.csv',
        help='Path to the input file'
    )
    parser.add_argument(
        '--n_rep',
        type=int,
        default=100,
        help='Number of repetitions with random initializations'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../data/coefficients_100runs.csv',
        help='Path to the output file'
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()

