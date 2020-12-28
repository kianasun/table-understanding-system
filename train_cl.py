import yaml
import sys
import numpy as np
from data_loader.load_majid_data import LoadCell2VecData
from cell_classifier.c2v.train_classifier import C2VClassifierTrainer
from cell_classifier.crf.train_grid_crf_v2 import GridCRFTrainer
from cell_classifier.mlp.train_classifier import MLPClassifierTrainer
from block_extractor.c2v.train_extractor import C2VExtractorTrainer
from block_extractor.crf.train_chain_crf_v2 import ChainCRFTrainer
from layout_detector.random_forest.train_classifier import RFClassifierTrainer
from layout_detector.crf.train_graph_crf import GraphCRFTrainer
import pandas as pd
import csv
import json
import random
import argparse
import os

def main(config, train_cc, train_be, train_lp):

    data_loader = LoadCell2VecData(config['jl_path'])

    output_dir = os.path.join(config["data_path"], config["dataset"])

    with open(os.path.join(output_dir, config["fold_file"]), 'r') as outfile:
        folds = json.load(outfile)

    folds_list = []

    for i, fold in enumerate(folds):
        #if i != 0:
        #    continue
        train_indices = fold["train"]
        dev_indices = fold["dev"]

        train_sheets, train_celltypes, train_blocktypes, train_layouttypes = data_loader.get_tables_from_indices(train_indices)
        dev_sheets, dev_celltypes, dev_blocktypes, dev_layouttypes = data_loader.get_tables_from_indices(dev_indices)

        result_path = os.path.join(config["model_path"], config["dataset"])
        os.makedirs(result_path, exist_ok=True)

        if train_cc:
            #model_path = os.path.join(result_path,
            #                config["c2v"]["cell_classifier_model_file"]+ str(i) +".model")
            #cc_trainer = C2VClassifierTrainer(model_path)
            #cc_trainer.fit(train_sheets, train_celltypes, dev_sheets, dev_celltypes)

            model_path = os.path.join(result_path,
                            config["crf"]["cell_classifier_model_file"]+ str(i) +".model")
            cc_trainer = GridCRFTrainer(model_path)
            cc_trainer.fit(train_sheets, train_celltypes, dev_sheets, dev_celltypes)

            #model_path = os.path.join(result_path,
            #                config["mlp"]["cell_classifier_model_file"]+ str(i) +".model")
            #cc_trainer = MLPClassifierTrainer(model_path)
            #cc_trainer.fit(train_sheets, train_celltypes, dev_sheets, dev_celltypes)

        if train_be:
            #model_path = os.path.join(result_path,
            #                config["c2v"]["block_extractor_model_file"]+ str(i) + ".model")
            #be_trainer = C2VExtractorTrainer(model_path)
            #be_trainer.fit(train_sheets, train_blocktypes, dev_sheets, dev_blocktypes)

            model_path = os.path.join(result_path,
                            config["crf"]["block_extractor_model_file"]+ str(i) + ".model")
            be_trainer = ChainCRFTrainer(model_path)
            be_trainer.fit(train_sheets, train_blocktypes, dev_sheets, dev_blocktypes)

        if train_lp:
            model_path = os.path.join(result_path,
                            config["crf"]["layout_predictor_model_file"]+ str(i) + ".model")
            lp_trainer = GraphCRFTrainer(model_path)
            lp_trainer.fit(train_sheets, train_layouttypes, dev_sheets, dev_layouttypes)

            model_path = os.path.join(result_path,
                            config["rf"]["layout_predictor_model_file"]+ str(i) + ".model")
            lp_trainer = RFClassifierTrainer(model_path)
            lp_trainer.fit(train_sheets, train_layouttypes, dev_sheets, dev_layouttypes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--cell', dest='cell', action='store_true')
    parser.add_argument('--block', dest='block', action='store_true')
    parser.add_argument('--layout', dest='layout', action='store_true')

    FLAGS, unparsed = parser.parse_known_args()

    with open(FLAGS.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    main(config, FLAGS.cell, FLAGS.block, FLAGS.layout)
