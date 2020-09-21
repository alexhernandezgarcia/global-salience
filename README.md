# Global visual salience of competing stimuli

By [Alex Hernandez-Garcia](https://alexhernandezgarcia.github.io/), Ricardo Ramos Gameiro, [Alessandro Grillini](https://www.rug.nl/staff/a.grillini/research) and [Peter König](https://scholar.google.com/citations?user=Ieubd0EAAAAJ&hl=en)

This repository contains the code used to compute and analyse the global visual salience of natural images from eye-tracking data, as described in the paper [Global visual salience of competing stimuli](https://psyarxiv.com/z7qp5/) ([Journal of Vision, 2020](https://jov.arvojournals.org/article.aspx?articleid=2770513)).

## Data

The complete data sets of this project are available on the [Supplementary material OSF page of the preprint](https://osf.io/t4gq9/). The following files are available:

* [`data_raw.mat`](https://osf.io/nsp3y/): The basic data set containing the eye-tracking data from the experimental sessions.
* [`data_all.csv`](https://osf.io/exkrp/): The most complete version of the data set, containing the original data from `data_raw.mat`, as well as other useful information.
* [`data_firstfixation.csv`](https://osf.io/8g2zp/): A derived data set containing data relative to the first fixations at each trial. Also available in this repository (`./data/data_firstfixation.csv`)

## Usage

### 1. Clone the repository:

```
git clone https://github.com/alexhernandezgarcia/global-salience.git
```

### 2. Install the package

#### `pip`

Run the following command from inside the directory:

```
pip install -e .
```

### 3.1 Try some of the [examples](./examples)

```
python ./examples/eval.py --input ./data/data_firstfixation.csv --target first --test_pct 0.2 --test_folds 25
```

### 3.2 Analyze the data as in the [Jupyter notebooks](./notebooks)

## Citation

If you use this code for scientific purposes, please cite:

*Alex Hernandez-Garcia, Ricardo Ramos Gameiro, Alessandro Grillini, Peter König, 2019. Global visual salience of competing stimuli. PsyArXiv:z7qp5*

	@article{hergar2019globalsalience,
		author = {Hernandez-Garcia, Alex and Ramos Gameiro, Ricardo and Grillini, Alessandro and K{\"o}nig, Peter},
		title = {Global visual salience of competing stimuli},
		journal = {Journal of Vision},
		year = {2020},
		volume = {20},
		number = {7},
		pages = {27-27},
		month = {07},
		doi = {10.1167/jov.20.7.27},
	}

## License

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
