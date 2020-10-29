import os
import random
import numpy as np
from reader.file_reader import get_file_reader
from config_psl import config, get_full_path
from type.cell.semantic_cell_type import SemanticCellType
from type.block.function_block_type import FunctionBlockType
from type.cell.cell_type_pmf import CellTypePMF
from type.block.block_type_pmf import BlockTypePMF
from type.block.simple_block import SimpleBlock
from type.layout.basic_edge_type import BasicEdgeType
from type.layout.layout_graph import LayoutGraph
from typing import List

class LoadDatagovData:
    def __init__(self):

        self.data_path = get_full_path(config['datagov']['source_data'])

        self.celltype_path = get_full_path(config['datagov']['celltype'])

        self.blocktype_path = get_full_path(config['datagov']['blocktype'])

        self.layouttype_path = get_full_path(config['datagov']['layouttype'])

        self.tables = self.load_files(get_full_path(config['datagov']['table_list']))

        self.celltypes, self.blocktypes, self.layouttypes = self.load_types_all_tables()

    def load_files(self, tab_list):

        tables = {}

        with open(tab_list, "r") as infile:

            for line in infile:

                tab_name, sheet_name = line.strip().split("\t")

                file_path = os.path.join(self.data_path, tab_name)

                if tab_name.endswith(".csv"):
                    sheet = get_file_reader(file_path).get_sheet_by_index(int(sheet_name))
                else:
                    sheet = get_file_reader(file_path).get_sheet_by_name(sheet_name)

                if sheet is None:

                    continue

                if tab_name not in tables:

                    tables[tab_name] = {}

                if sheet_name not in tables[tab_name]:

                    tables[tab_name][sheet_name] = sheet

        return tables

    def read_type_a_table(self, tab_name, tab_sheet, path):

        #max_row, max_col = 0, 0

        types = []

        for f in os.listdir(path):

            abs_path = os.path.join(path, f)

            if not (os.path.isfile(abs_path) and f.startswith("{}_{}".format(tab_name, tab_sheet))):

                continue

            typ = f.split("_")[-1].split(".")[0]

            with open(abs_path, "r") as infile:

                for line in infile:

                    lx, ly, rx, ry = [int(_) for _ in line.strip().split()]

                    #max_row = max(max_row, rx)

                    #max_col = max(max_col, ry)

                    types.append((lx, ly, rx, ry, typ))

        #return types, max_row, max_col
        return types


    def load_celltype_a_table(self, tab_name, tab_sheet):

        #celltypes, max_row, max_col = self.read_type_a_table(tab_name, tab_sheet, self.celltype_path)
        celltypes = self.read_type_a_table(tab_name, tab_sheet, self.celltype_path)

        values = self.tables[tab_name][tab_sheet].values

        n, m = values.shape
        
        tags = np.empty((n, m), dtype=CellTypePMF)

        for (lx, ly, rx, ry, typ) in celltypes:

            for i in range(lx, min(rx + 1, n)):

                for j in range(ly, min(ry + 1, m)):

                    if len(values[i][j]) == 0:
                        tags[i][j] =  CellTypePMF({SemanticCellType.EMPTY: 1})

                    elif values[i][j].lower() in ["nan", "unknown", "-", "NA", "NA**"]:

                        tags[i][j] = CellTypePMF({SemanticCellType.EMPTY: 1})

                    else:
                        tags[i][j] = CellTypePMF({SemanticCellType.inverse_dict[typ]: 1})


        return tags


    def load_blocktype_a_table(self, tab_name, tab_sheet):

        blocktypes = self.read_type_a_table(tab_name, tab_sheet, self.blocktype_path)
        
        values = self.tables[tab_name][tab_sheet].values
        
        n, m = values.shape
        
        blocks = []

        blocks_dict = {}

        for (lx, ly, rx, ry, typ) in blocktypes:

            blktyp = BlockTypePMF({FunctionBlockType.inverse_dict[typ]: 1})

            blocks.append(SimpleBlock(blktyp, ly, min(ry, m-1), lx, min(rx, n-1) ))

            blocks_dict[(lx, ly)] = len(blocks) - 1

        return blocks, blocks_dict

    def load_layouttype_a_table(self, tab_name, tab_sheet, blocks, block_dict):

        layout_graph = LayoutGraph(blocks)

        for f in os.listdir(self.layouttype_path):

            abs_path = os.path.join(self.layouttype_path, f)

            if not (os.path.isfile(abs_path) and f.startswith("{}_{}".format(tab_name, tab_sheet))):

                continue

            with open(abs_path, "r") as infile:

                for line in infile:

                    temp = line.strip().split()

                    lx, ly, rx, ry = [int(_) for _ in temp[:-1]]

                    typ = temp[-1]

                    layout_graph.add_edge(BasicEdgeType.str_to_edge_type[typ],
                                          block_dict[(lx, ly)], block_dict[(rx, ry)])

        return layout_graph

    def load_types_all_tables(self):

        celltypes, blocktypes, layouttypes = {}, {}, {}

        for k in self.tables:

            celltypes[k] = {}
            blocktypes[k] = {}
            layouttypes[k] = {}

            for kk in self.tables[k]:
                celltypes[k][kk] = self.load_celltype_a_table(k, kk)

                blocktypes[k][kk], temp_dict = self.load_blocktype_a_table(k, kk)

                layouttypes[k][kk] = self.load_layouttype_a_table(k, kk, blocktypes[k][kk],
                                                                  temp_dict)

        return celltypes, blocktypes, layouttypes


    def split_tables(self, k=5, seed=0):

        temp = [_ for _ in self.tables]

        random.seed(seed)

        random.shuffle(temp)

        indices = []

        each_len = int(len(temp) / k)

        for i in range(k):

            indices.append(temp[i*each_len : (i+1)*each_len])

        return indices

    def get_tables_from_indices(self, indices):

        sheet_list, celltype_list, blocktype_list, layouttype_list = [], [], [], []
        #print([(k, s) for k in indices for s in self.tables[k]])

        for k in indices:

            for s in self.tables[k]:

                sheet_list.append(self.tables[k][s])

                celltype_list.append(self.celltypes[k][s])

                blocktype_list.append(self.blocktypes[k][s])

                layouttype_list.append(self.layouttypes[k][s])

        return sheet_list, celltype_list, blocktype_list, layouttype_list
