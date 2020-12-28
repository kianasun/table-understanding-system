import yaml
import sys
import json
import os
import argparse
import numpy as np
from type.cell.semantic_cell_type import SemanticCellType
from sklearn.metrics import f1_score
from utils.convert_utils import convert_str_id_cc

def main(json_file):

    with open(json_file, 'r') as outfile:
        cc_tags = json.load(outfile)

    id2str = {v.id(): k for k, v in SemanticCellType.inverse_dict.items()}
    labels = [id2str[_] for _ in range(len(id2str))]

    scores = []
    macro_scores = []
    time_list = []

    for i, tags in enumerate(cc_tags):
        gt = convert_str_id_cc(tags["gt"])
        pred = convert_str_id_cc(tags["predict"])
        #print(len(tags["gt"]))
        time_list.append(tags["time"] / float(len(tags["gt"])))

        scores.append(f1_score(gt, pred, average=None))
        macro_scores.append(f1_score(gt, pred, average="macro"))

    print("stddev", np.std(np.array(macro_scores)))
    scores = np.mean(np.array(scores), axis=0)
    print("\n".join(["{}: {}".format(labels[i], scores[i]) for i in range(len(labels))]))
    print("macro avg: {}".format(sum(macro_scores) / len(macro_scores)))
    print("time avg: {} seconds".format(sum(time_list) / len(time_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS.input_file)
