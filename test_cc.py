import yaml
import sys
import argparse
from data_loader.load_majid_data import LoadCell2VecData
from cell_classifier.psl_cell_classifier import PSLCellClassifier
from cell_classifier.c2v_cell_classifier_v2 import C2VCellClassifierV2
import json
import os

def predict_one_fold(model_file, config, method, sheet_list, celltypes):

    if method == "psl":
        classifier = PSLCellClassifier(model_file, config)
    else:
        classifier = C2VCellClassifierV2(model_file)

    tags = classifier.classify_cells_all_tables(sheet_list)

    ret = [[[tag[r][c].get_best_type().str() for c in range(tag.shape[1])]
            for r in range(tag.shape[0])] for tag in tags]

    gt = [[[None if tag[r][c] is None else tag[r][c].get_best_type().str()
            for c in range(tag.shape[1])]
                for r in range(tag.shape[0])] for tag in celltypes]

    return {"predict": ret, "gt": gt}


def main(config, method):

    data_loader = LoadCell2VecData(config['jl_path'])

    output_dir = os.path.join(config["data_path"], config["dataset"])

    with open(os.path.join(output_dir, config["fold_file"]), 'r') as outfile:
        folds = json.load(outfile)

    pred_list = []

    result_path = os.path.join(config["model_path"], config["dataset"])

    for i, fold in enumerate(folds):
        test_indices = fold["eval"]

        test_sheets, test_celltypes, _, __ = data_loader.get_tables_from_indices(test_indices)

        if config["dataset"] == "dg":
            model_path = os.path.join(result_path,
                            config["c2v"]["cell_classifier_model_file"] + str(i) +".model")
        else:
            model_path = config["c2v"]["cell_classifier_model_file"] + ".model"


        pred = predict_one_fold(model_path, config, method, test_sheets, test_celltypes)

        pred_list.append(pred)

    with open(os.path.join(result_path, config[method]["cc_output"]), 'w+') as outfile:
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
