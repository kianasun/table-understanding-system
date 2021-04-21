import yaml
import sys
import numpy as np
from data_loader.load_majid_data import LoadCell2VecData
import pandas as pd
import csv
import json
import random
import argparse
import os

def main(config):

    data_loader = LoadCell2VecData(config['jl_path'])

    indices = data_loader.split_tables(k=config['num_of_folds'])

    all_indices = [item for lst in indices for item in lst]

    # We remove the first 10 tables which were used for rule development
    all_indices = all_indices[10:]

    indices = data_loader.split_indices(all_indices, k=config['num_of_folds'])

    # set a seed for splitting train/dev
    random.seed(config['seed'])

    folds_list = []

    for i in range(config['num_of_folds']):

        other_indices = [_ for idx in indices[:i] + indices[i+1:] for _ in idx]

        random.shuffle(other_indices)

        # 9:1 for splitting train/dev
        split_point = int(len(other_indices)*0.9)

        folds_list.append({"train": other_indices[:split_point],
                           "dev": other_indices[split_point:],
                           "eval": indices[i]})

    output_dir = os.path.join(config["data_path"], config["dataset"])
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, config["fold_file"]), 'w+') as outfile:
        json.dump(folds_list, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    FLAGS, unparsed = parser.parse_known_args()

    with open(FLAGS.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    main(config)
