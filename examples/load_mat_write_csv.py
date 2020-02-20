import argparse

from globalsalience.utils import load_data

# Initialize the Flags container
FLAGS = None

def main(argv=None):

    df = load_data(FLAGS.input, 
                   mat_variable_name=FLAGS.mat_variable_name,
                   categories_yml=FLAGS.categories_yml,
                   output_csv=FLAGS.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        type=str,
        default='../data/data_raw.mat',
        help='Path to the input file'
    )
    parser.add_argument(
        '--mat_variable_name',
        type=str,
        default='fixmat',
        help='Name of the variable to be retrieved as data from the .mat file'
    )
    parser.add_argument(
        '--categories_yml',
        type=str,
        default='../data/categories.yml',
        help='Path to the YAML file that specifies the category of each image'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Name of the output CSV file'
    )

    FLAGS, unparsed = parser.parse_known_args()
    main()

