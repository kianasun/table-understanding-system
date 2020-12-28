import yaml
import sys
import json
import os
import argparse
import numpy as np
from type.block.function_block_type import FunctionBlockType
from data_loader.load_majid_data import LoadCell2VecData
from sklearn.metrics import f1_score
from utils.convert_utils import convert_blk_celltype_id, convert_blk_blkobj

def get_cl_score(gt, pred):
    new_gt, new_pred = [], []
    for i in range(len(gt)):
        if gt[i] == -1:
            continue
        new_gt.append(gt[i])
        new_pred.append(pred[i])

    return f1_score(new_gt, new_pred, average=None),\
            f1_score(new_gt, new_pred, average="macro")

def assign_blk_to_cell(blks, sheet):
    r, c = sheet.values.shape
    assignment = np.zeros((r, c), dtype=int)
    assignment.fill(-1)
    for idx, blk in enumerate(blks):
        lx, ly = blk.top_row, blk.left_col
        rx, ry = blk.bottom_row, blk.right_col

        for i in range(lx, rx + 1):
            for j in range(ly, ry + 1):
                assignment[i][j] = idx
    return assignment

def eob(blk1, blk2):
    blk1_lx, blk1_ly = blk1.top_row, blk1.left_col
    blk1_rx, blk1_ry = blk1.bottom_row, blk1.right_col

    blk2_lx, blk2_ly = blk2.top_row, blk2.left_col
    blk2_rx, blk2_ry = blk2.bottom_row, blk2.right_col

    temp = abs(blk1_lx - blk2_lx)
    temp = max(temp, abs(blk1_ly - blk2_ly))
    temp = max(temp, abs(blk1_rx - blk2_rx))
    temp = max(temp, abs(blk1_ry - blk2_ry))

    area = blk1.get_intersecting_area(blk2)

    return temp / float(area)

def evaluate_eob(sheets, gt, pred):
    gt_obj, pred_obj = convert_blk_blkobj(gt), convert_blk_blkobj(pred)

    all_score = []
    total_score = []

    for idx, sheet in enumerate(sheets):
        r, c = sheet.values.shape
        truth_assign = assign_blk_to_cell(gt_obj[idx], sheet)
        pred_assign = assign_blk_to_cell(pred_obj[idx], sheet)

        scores = []

        for i in range(r):
            for j in range(c):
                if truth_assign[i][j] == -1:
                    continue

                truth_blk = gt_obj[idx][truth_assign[i][j]]
                pred_blk = pred_obj[idx][pred_assign[i][j]]
                eob_score = eob(truth_blk, pred_blk)

                #if (truth_assign[i][j], pred_assign[i][j]) not in scores:
                #    scores[(truth_assign[i][j], pred_assign[i][j])] = [eob_score]
                #else:
                #    scores[(truth_assign[i][j], pred_assign[i][j])].append(eob_score)
                scores.append(eob_score)

        #total_score.append(sum([sum(v) / len(v) for k, v in scores.items()]))
        total_score.append(sum(scores))

    return sum(total_score) / len(total_score)


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
    eob_scores = []
    time_list = []

    for i, fold in enumerate(folds):
        test_indices = fold["eval"]

        test_sheets, _, __, ___ = data_loader.get_tables_from_indices(test_indices)

        gt = convert_blk_celltype_id(pred_blocks[i]["gt"], test_sheets)
        pred = convert_blk_celltype_id(pred_blocks[i]["predict"], test_sheets)

        temp_all, temp_macro = get_cl_score(gt, pred)
        scores.append(temp_all)
        macro_scores.append(temp_macro)

        eob_scores.append(evaluate_eob(test_sheets, pred_blocks[i]["gt"],
                                       pred_blocks[i]["predict"]))
        print(len(test_sheets))
        time_list.append(pred_blocks[i]["time"] / float(len(test_sheets)))

        #break

    print(scores)
    print("stddev", np.std(np.array(macro_scores)))
    scores = np.mean(np.array(scores), axis=0)
    print("\n".join(["{}: {}".format(labels[i], scores[i]) for i in range(len(labels))]))
    print("macro avg: {}".format(sum(macro_scores) / len(macro_scores)))
    print("eob avg: {}".format(sum(eob_scores) / len(eob_scores)))
    print("time avg: {} seconds".format(sum(time_list) / len(time_list)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str)
    parser.add_argument('--config', type=str)
    FLAGS, unparsed = parser.parse_known_args()

    with open(FLAGS.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)
    main(config, FLAGS.input_file)
