import argparse

from globalsalience.scorer import PairwiseComparisonsScorer

# Initialize the Flags container
FLAGS = None

def main(argv=None):

    # Initialize the scorer and load the first fixations data set
    scorer = PairwiseComparisonsScorer(target='first')
    scorer.load_data(FLAGS.input, compute_features=False, do_filter=False)

    # K-fold evaluation
    print('\nAverage performance across {:d} folds\n'.format(FLAGS.test_folds))
    scorer.kfold_evaluation(test_pct=FLAGS.test_pct, 
                            test_folds=FLAGS.test_folds, do_print=True)
    print('\nPerformance of a random baseline model\n')
    scorer.kfold_evaluation(test_pct=FLAGS.test_pct, 
                            test_folds=FLAGS.test_folds, shuffle_labels=True, 
                            do_print=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        default='../data/data_firstfixation.csv',
        help='Path to the input file'
    )
    parser.add_argument(
        '--test_pct',
        type=float,
        default=0.2,
        help='The percentage of data in the test partitions'
    )
    parser.add_argument(
        '--test_folds',
        type=int,
        default=25,
        help='The percentage of data in the test partitions'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()

