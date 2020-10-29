import yaml
import sys
import json
import os
import argparse
import numpy as np
from data_loader.load_majid_data import LoadCell2VecData
from cell_classifier.psl_cell_classifier import PSLCellClassifier
from block_extractor.block_extractor_psl_v2 import BlockExtractorPSLV2
from block_extractor.block_extractor_c2v import BlockExtractorC2V
from block_extractor.block_extractor_c2v_pretrain import BlockExtractorC2VPretrain
from type.cell.cell_type_pmf import CellTypePMF
from type.cell.semantic_cell_type import SemanticCellType
from utils.convert_utils import convert_str_np_cc, convert_blkobj_celltype_id,\
                                convert_blk_celltype_id
from evaluate_be import get_cl_score
import itertools

def validate_psl(cc_model_file, be_model_file, config, eval_sheets, eval_blocks):

    cc_classifier = PSLCellClassifier(cc_model_file, config)
    cc_tags = cc_classifier.classify_cells_all_tables(eval_sheets)

    beta = [0.01, 0.05]
    lmd = [5, 10]
    best_lmd = None
    best_beta = None
    best_score = 0
    combs = [beta, lmd]
    for para in list(itertools.product(*combs)):
        (b, l) = para
        classifier = BlockExtractorPSLV2(be_model_file, config, beta=b, lmd=l)

        ret = classifier.extract_blocks_all_tables(eval_sheets, cc_tags)

        gt = convert_blkobj_celltype_id(eval_blocks, eval_sheets)
        pred = convert_blkobj_celltype_id(ret, eval_sheets)
        score = get_cl_score(gt, pred)[1]
        print("beta", b, "lmd", l, "score", score)

        if score > best_score:
            best_score = score
            best_lmd = l
            best_beta = b

    return {"beta": best_beta, "lmd": best_lmd}


def predict_one_fold(cc_model_file, be_model_file, config, method,
                     test_tup, eval_tup):
    test_sheets, test_celltypes, test_blocktypes = test_tup
    eval_sheets, eval_blocktypes = eval_tup

    if method == "psl":
        par_dict = validate_psl(cc_model_file, be_model_file, config,
                                eval_sheets, eval_blocktypes)
        print("best par_dict", par_dict)
        classifier = BlockExtractorPSLV2(be_model_file, config, beta=par_dict["beta"],
                                         lmd=par_dict["lmd"])
    elif ("use_rnn" in config["block_extractor"]) and config["block_extractor"]["use_rnn"]:
        classifier = BlockExtractorC2VPretrain(be_model_file, config)
    else:
        classifier = BlockExtractorC2V(be_model_file)

    pred_block_list = classifier.extract_blocks_all_tables(test_sheets, test_celltypes)

    ret = [[(blk.top_row, blk.left_col, blk.bottom_row, blk.right_col,
                blk.block_type.get_best_type().str())
                for blk in blocks] for blocks in pred_block_list]

    gt = [[(blk.top_row, blk.left_col, blk.bottom_row, blk.right_col,
                blk.block_type.get_best_type().str())
                for blk in blocks] for blocks in test_blocktypes]

    print("score:", get_cl_score(convert_blk_celltype_id(gt, test_sheets),
                                 convert_blk_celltype_id(ret, test_sheets)))

    return {"predict": ret, "gt": gt}


def main(config, method):

    data_loader = LoadCell2VecData(config['jl_path'])

    output_dir = os.path.join(config["data_path"], config["dataset"])

    result_path = os.path.join(config["model_path"], config["dataset"])

    with open(os.path.join(output_dir, config["fold_file"]), 'r') as outfile:
        folds = json.load(outfile)

    with open(os.path.join(result_path, config["psl"]["cc_output"]), 'r') as outfile:
        cc_tags = json.load(outfile)

    pred_list = []

    for i, fold in enumerate(folds):
        test_indices = fold["eval"]
        dev_indices = fold["dev"]

        cc_pred = convert_str_np_cc(cc_tags[i]["predict"])

        test_sheets, _, test_blocktypes, __ = data_loader.get_tables_from_indices(test_indices)

        dev_sheets, _, dev_blocktypes, __ = data_loader.get_tables_from_indices(dev_indices)

        if ("use_rnn" in config["block_extractor"]) and config["block_extractor"]["use_rnn"]:
            be_model_path = config["c2v"]["cl_model"] + str(i) + ".model"
        else:
            be_model_path = os.path.join(result_path,
                            config["c2v"]["block_extractor_model_file"]+ str(i) +".model")

        if config["dataset"] == "dg":
            cc_model_path = os.path.join(result_path,
                            config["c2v"]["cell_classifier_model_file"] + str(i) +".model")
        else:
            cc_model_path = config["c2v"]["cell_classifier_model_file"] + ".model"


        pred = predict_one_fold(cc_model_path, be_model_path, config, method,
                               (test_sheets, cc_pred, test_blocktypes),
                               (dev_sheets, dev_blocktypes))

        pred_list.append(pred)
        #break

    with open(os.path.join(result_path, config[method]["be_output"]), 'w+') as outfile:
        json.dump(pred_list, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--method', type=str)
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.method not in ["psl", "c2v"]:
        print("Only supports PSL or RandomForest")
        sys.exit(0)

    with open(FLAGS.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)

    main(config, FLAGS.method)
