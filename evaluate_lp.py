import yaml
import sys
import json
import os
import argparse
import numpy as np
from type.block.function_block_type import FunctionBlockType
from data_loader.load_majid_data import LoadCell2VecData
from sklearn.metrics import f1_score
from utils.convert_utils import convert_blk_blkobj, convert_lst_graphobj
from type.layout.basic_edge_type import BasicEdgeType

def generate_matrix(graph, idx_map, num):
    res = np.array([[BasicEdgeType.str_to_edge_type["empty"].id()
                for _ in range(num)] for __ in range(num)])
    temp = {}

    for i, edges in enumerate(graph.inEdges):
        for (_type, _from) in edges:
            if (idx_map[_from], idx_map[i]) not in temp:
                temp[(idx_map[_from], idx_map[i])] = {}
            if _type.id() not in temp[(idx_map[_from], idx_map[i])]:
                temp[(idx_map[_from], idx_map[i])][_type.id()] = 0

            temp[(idx_map[_from], idx_map[i])][_type.id()] += 1

    for (i, j) in temp:
        if i != -1 and j != -1:
            res[i][j] = sorted(temp[(i, j)].items(), key=lambda x: x[1],
                        reverse=True)[0][0]

    return res

def pred2truth(pred, truth):
    p2t = []

    for i in range(len(pred.nodes)):

        idx, max_inter = -1, 0

        for j in range(len(truth.nodes)):
            temp = truth.nodes[j].get_intersecting_area(pred.nodes[i])
            if temp > max_inter:
                max_inter = temp
                idx = j

        p2t.append(idx)

    return p2t

def get_cl_score(preds, truths):
    all_pred = np.array([])
    all_truth = np.array([])

    for i in range(len(preds)):
        p2t = pred2truth(preds[i], truths[i])
        num = len(truths[i].nodes)

        pred_list = generate_matrix(preds[i], p2t, num)
        truth_list = generate_matrix(truths[i], {j:j for j in range(len(truths[i].nodes))},
                                    num)

        pred_list = pred_list.flatten()
        truth_list = truth_list.flatten()

        all_pred = np.concatenate((all_pred, pred_list))
        all_truth = np.concatenate((all_truth, truth_list))

    return f1_score(all_truth, all_pred, average=None),\
            f1_score(all_truth, all_pred, average="macro")


def main(config, json_file):
    data_loader = LoadCell2VecData(config['jl_path'])

    output_dir = os.path.join(config["data_path"], config["dataset"])

    result_path = os.path.join(config["model_path"], config["dataset"])

    with open(os.path.join(output_dir, config["fold_file"]), 'r') as outfile:
        folds = json.load(outfile)

    with open(os.path.join(result_path, config["psl"]["be_output"]), 'r') as outfile:
        be_tags = json.load(outfile)

    with open(json_file, 'r') as outfile:
        preds = json.load(outfile)

    id2str = {v.id(): k for k, v in BasicEdgeType.str_to_edge_type.items()}
    labels = [id2str[_] for _ in range(len(id2str))]
    scores = []
    time_list = []
    macro_scores = []

    for i, fold in enumerate(folds):
        test_indices = fold["eval"]

        #test_sheets, _, __, ___ = data_loader.get_tables_from_indices(test_indices)

        gt = convert_lst_graphobj(preds[i]["gt"], convert_blk_blkobj(be_tags[i]["gt"]))

        pred = convert_lst_graphobj(preds[i]["predict"], convert_blk_blkobj(be_tags[i]["predict"]))

        temp_all, temp_macro = get_cl_score(pred, gt)
        scores.append(temp_all)
        macro_scores.append(temp_macro)
        time_list.append(preds[i]["time"] / float(len(test_indices)))
        #break

    #print(scores)

    print("stddev", np.std(np.array(macro_scores)))
    scores = np.mean(np.array(scores), axis=0)
    print("\n".join(["{}: {}".format(labels[i], scores[i]) for i in range(len(labels))]))
    print("macro avg: {}".format(sum(macro_scores) / len(macro_scores)))
    print("time avg: {} seconds".format(sum(time_list) / len(time_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--config', type=str)
    FLAGS, unparsed = parser.parse_known_args()

    with open(FLAGS.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)
    main(config, FLAGS.input_file)
