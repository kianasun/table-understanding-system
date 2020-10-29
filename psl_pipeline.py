import yaml
import sys
import pyexcel as pyx
import xlrd
import numpy as np
from data_loader.load_majid_data import LoadCell2VecData
from cell_classifier.psl_cell_classifier import PSLCellClassifier
from cell_classifier.c2v_cell_classifier import C2VCellClassifier
from block_extractor.block_extractor_psl_v2 import BlockExtractorPSLV2
from layout_detector.layout_detector_psl import LayoutDetectorPSL
import pandas as pd
import csv

def main():

    data_loader = LoadCell2VecData()

    indices = data_loader.split_tables(k=50)

    sheet_list, celltype_list, blocktype_list, layouttype_list = data_loader.get_tables_from_indices(indices[0])
    print(sheet_list[2].values)

    cell_classifier = C2VCellClassifier()

    c2v_tags = cell_classifier.classify_cells_all_tables(sheet_list)

    cc_classifier = PSLCellClassifier()

    tags = cc_classifier.classify_cells_all_tables(sheet_list, None)

    extractor = BlockExtractorPSLV2()

    blocks = extractor.extract_blocks_all_tables(sheet_list, tags, c2v_tags)
    for i, blk in enumerate(blocks[2]):
        print(i, blk)

    detector = LayoutDetectorPSL()

    layouts = detector.detect_layout(sheet_list[2], tags[2], blocks[2])

    layouts.print_layout()

    with open('./test_sheet.csv', 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',')
        for row in c2v_tags[2]:
            spamwriter.writerow([r.get_best_type().str() for r in row])

        for row in tags[2]:
            spamwriter.writerow([r.get_best_type().str() for r in row])
        #spamwriter.writerows(sheet_list[2].values)

if __name__ == "__main__":
    main()
