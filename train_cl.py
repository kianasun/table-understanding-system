import yaml
import sys
import numpy as np
from data_loader.load_majid_data import LoadCell2VecData
from cell_classifier.c2v.train_classifier import C2VClassifierTrainer
from block_extractor.c2v.train_extractor import C2VExtractorTrainer
import pandas as pd
import csv
import json
import random
import argparse
import os

def main(config, train_cc, train_be):

    data_loader = LoadCell2VecData(config['jl_path'])

    output_dir = os.path.join(config["data_path"], config["dataset"])

    with open(os.path.join(output_dir, config["fold_file"]), 'r') as outfile:
        folds = json.load(outfile)

    folds_list = []

    for i, fold in enumerate(folds):
        train_indices = fold["train"]
        dev_indices = fold["dev"]

        train_sheets, train_celltypes, train_blocktypes, __ = data_loader.get_tables_from_indices(train_indices)
        dev_sheets, dev_celltypes, dev_blocktypes, __ = data_loader.get_tables_from_indices(dev_indices)

        result_path = os.path.join(config["model_path"], config["dataset"])
        os.makedirs(result_path, exist_ok=True)

        if train_cc:
            model_path = os.path.join(result_path,
                            config["c2v"]["cell_classifier_model_file"]+ str(i) +".model")
            cc_trainer = C2VClassifierTrainer(model_path)
            cc_trainer.fit(train_sheets, train_celltypes, dev_sheets, dev_celltypes)

        if train_be:
            model_path = os.path.join(result_path,
                            config["c2v"]["block_extractor_model_file"]+ str(i) + ".model")
            be_trainer = C2VExtractorTrainer(model_path)
            be_trainer.fit(train_sheets, train_blocktypes, dev_sheets, dev_blocktypes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--cell', dest='cell', action='store_true')
    parser.add_argument('--block', dest='block', action='store_true')

    FLAGS, unparsed = parser.parse_known_args()

    with open(FLAGS.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    main(config, FLAGS.cell, FLAGS.block)
