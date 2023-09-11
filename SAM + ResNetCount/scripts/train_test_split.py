import os, shutil, glob
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

def make_subset(subset, df, dest_dir, target_csv):
    """
    Helper function to transfer files and make corresponding target csv for a split
    Args:
        subset (list): list of ids
        df (pandas.Dataframe): dataframe consisting of same list of ids and target values
        dest_dir (string): destination directory
        target_csv (string): csv file name
    """
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    subset_ids = [file.split('/')[-1] for file in subset]

    subset_df = df[df['id'].isin(subset_ids)]

    for path in [s + '*' for s in subset]:
        for file in glob.glob(path):
            shutil.copy(file, dest_dir)

    subset_df.to_csv(os.path.join(dest_dir, target_csv), index=False, header=False)

def main(args):
    root_dir = args.root_dir
    train_dir = os.path.join(root_dir, 'train')
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    test_dir = os.path.join(root_dir, 'test')
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)


    # csv file containing target values
    target_csv = 'Leaf_counts.csv'

    # append all plant names to a lists
    types = os.listdir(root_dir)
    types.remove('train')
    types.remove('test')
    for type in types:
        type_dir = os.path.join(root_dir, type)
        plant_names = [os.path.join(type_dir, '_'.join(name.split('_')[:-1])) for name in os.listdir(type_dir) if not name[0].isupper()]

        # remove duplicates
        plant_names = list(dict.fromkeys(plant_names))
        df = pd.read_csv(os.path.join(type_dir, target_csv), names=['id', 'count'])

        train, test = train_test_split(plant_names, test_size=0.3, random_state=33)
        make_subset(train, df, os.path.join(train_dir, type), target_csv)
        make_subset(test, df, os.path.join(test_dir, type), target_csv)

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, help='absolute directory of data')

args = parser.parse_args()
main(args)