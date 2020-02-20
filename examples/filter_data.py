import argparse

from globalsalience.scorer import PairwiseComparisonsScorer

# Initialize the Flags container
FLAGS = None

def main(argv=None):

    scorer = PairwiseComparisonsScorer(target=FLAGS.target)
    scorer.load_data(FLAGS.input, compute_features=FLAGS.input_is_raw, 
                     do_filter=True)
    if FLAGS.output:
        scorer.data_df.to_csv(FLAGS.output, index_label='index')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        default='../data/data_all.csv',
        help='Path to the input file'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='first',
        help='The target variable for the analysis of information that must ' \
             'be filtered and retrieved'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Name of the output CSV file'
    )
    parser.add_argument(
        '--input_is_raw',
        action='store_true',
        dest='input_is_raw',
        help='If True, the features are recomputed'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()

