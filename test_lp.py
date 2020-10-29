import yaml
import sys
import json
import os
import argparse
import numpy as np
from type.block.function_block_type import FunctionBlockType
from data_loader.load_majid_data import LoadCell2VecData
from utils.convert_utils import convert_blk_blkobj, convert_str_id_cc
from layout_detector.layout_detector_psl import LayoutDetectorPSL

def predict_one_fold(config, method, test_tup):
    test_sheets, test_celltypes, test_blocktypes, test_layouts = test_tup

    if method == "psl":
        detector = LayoutDetectorPSL(config)

    layouts = detector.detect_layout_all_tables(test_sheets, test_celltypes, test_blocktypes)

    ret = [[(from_idx, i, typ.str()) for i in range(len(layout.nodes))
                for (typ, from_idx) in layout.inEdges[i]]
            for layout in layouts]

    gt = [[(from_idx, i, typ.str()) for i in range(len(layout.nodes))
                for (typ, from_idx) in layout.inEdges[i]]
            for layout in test_layouts]

    return {"predict": ret, "gt": gt}


def main(config, method):
    data_loader = LoadCell2VecData(config['jl_path'])

    output_dir = os.path.join(config["data_path"], config["dataset"])

    result_path = os.path.join(config["model_path"], config["dataset"])

    with open(os.path.join(output_dir, config["fold_file"]), 'r') as outfile:
        folds = json.load(outfile)

    with open(os.path.join(result_path, config["psl"]["be_output"]), 'r') as outfile:
        be_tags = json.load(outfile)

    with open(os.path.join(result_path, config["psl"]["cc_output"]), 'r') as outfile:
        cc_tags = json.load(outfile)

    pred_list = []
    for i, fold in enumerate(folds):
        print("fold {}".format(i))
        test_indices = fold["eval"]

        be_pred = convert_blk_blkobj(be_tags[i]["predict"])

        cc_pred= convert_str_id_cc(cc_tags[i]["predict"])

        test_sheets, test_celltypes, test_blocktypes, test_layouts = data_loader.get_tables_from_indices(test_indices)

        pred = predict_one_fold(config, method,
                               (test_sheets, cc_pred, be_pred, test_layouts))

        pred_list.append(pred)

    with open(os.path.join(result_path, config[method]["lp_output"]), 'w+') as outfile:
        json.dump(pred_list, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--config', type=str)
    FLAGS, unparsed = parser.parse_known_args()

    with open(FLAGS.config, 'r') as ymlfile:
        config = yaml.load(ymlfile)
    main(config, FLAGS.method)
