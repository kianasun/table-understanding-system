import yaml
import sys
import json
import os
import argparse
import numpy as np
from type.block.function_block_type import FunctionBlockType
from data_loader.load_majid_data import LoadCell2VecData
from sklearn.metrics import f1_score
from utils.convert_utils import convert_blk_celltype_id

def get_cl_score(gt, pred):
    new_gt, new_pred = [], []
    for i in range(len(gt)):
        if gt[i] == -1:
            continue
        new_gt.append(gt[i])
        new_pred.append(pred[i])

    return f1_score(new_gt, new_pred, average=None),\
            f1_score(new_gt, new_pred, average="macro")


def main(config, json_file):
    data_loader = LoadCell2VecData(config['jl_path'])

    output_dir = os.path.join(config["data_path"], config["dataset"])

    with open(os.path.join(output_dir, config["fold_file"]), 'r') as outfile:
        folds = json.load(outfile)

    with open(json_file, 'r') as outfile:
        pred_blocks = json.load(outfile)

    id2str = {v.id(): k for k, v in FunctionBlockType.inverse_dict.items()}
    labels = [id2str[_] for _ in range(len(id2str))]
    scores = []
    macro_scores = []

    for i, fold in enumerate(folds):
        test_indices = fold["eval"]

        test_sheets, _, __, ___ = data_loader.get_tables_from_indices(test_indices)

        gt = convert_blk_celltype_id(pred_blocks[i]["gt"], test_sheets)
        pred = convert_blk_celltype_id(pred_blocks[i]["predict"], test_sheets)

        temp_all, temp_macro = get_cl_score(gt, pred)
        scores.append(temp_all)
        macro_scores.append(temp_macro)
        #break

    print(scores)

    scores = np.mean(np.array(scores), axis=0)
    print("\n".join(["{}: {}".format(labels[i], scores[i]) for i in range(len(labels))]))
    print("macro avg: {}".format(sum(macro_scores) / len(macro_scores)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--config', type=str)
    FLAGS, unparsed = parser.parse_known_args()

    with open(FLAGS.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)
    main(config, FLAGS.input_file)
