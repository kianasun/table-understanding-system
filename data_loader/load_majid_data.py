import os
import random
import numpy as np
import json
from reader.file_reader import get_file_reader
from reader.sheet import Sheet
from type.cell.function_cell_type import FunctionCellType
from type.cell.semantic_cell_type import SemanticCellType
from type.block.function_block_type import FunctionBlockType
from type.block.function_block_type_v2 import FunctionBlockTypeV2
from type.layout.layout_graph import LayoutGraph
from type.layout.basic_edge_type import BasicEdgeType
from type.cell.cell_type_pmf import CellTypePMF
from type.block.block_type_pmf import BlockTypePMF
from type.block.simple_block import SimpleBlock
from type.layout.basic_edge_type import BasicEdgeType
from type.layout.layout_graph import LayoutGraph
from typing import List
import itertools

class LoadCell2VecData:
    def __init__(self, data_path):

        self.load_data(data_path)

    def construct_layout_graph(self, blocks, layouts):

        layout_graph = LayoutGraph(blocks)

        for (b1_lx, b1_ly, b1_rx, b1_ry, b2_lx, b2_ly, b2_rx, b2_ry, typ) in layouts:

            b1_set, b2_set = set(), set()

            if typ == "global":
                typ = "global_attribute"
            elif typ == "subset":
                typ = "supercategory"

            an_type = BasicEdgeType.str_to_edge_type[typ]

            for i, blk in enumerate(blocks):
                lx, ly = blk.get_top_row(), blk.get_left_col()

                rx, ry = blk.get_bottom_row(), blk.get_right_col()

                if b1_lx <= lx and b1_ly <= ly and lx <= b1_rx and ly <= b1_ry:

                    b1_set.add(i)

                elif b2_lx <= lx and b2_ly <= ly and lx <= b2_rx and ly <= b2_ry:

                    b2_set.add(i)

            for t_b1, t_b2 in itertools.product(list(b1_set), list(b2_set)):

                if typ == "supercategory":
                    layout_graph.add_edge(an_type, t_b2, t_b1)
                else:
                    layout_graph.add_edge(an_type, t_b1, t_b2)

        return layout_graph

    def load_data(self, filename):

        self.tables = []

        self.celltypes, self.blocktypes, self.layouttypes = [], [], []

        with open(filename, "r") as f:

            for line in f:

                dic = json.loads(line)

                table_id = dic["table_id"].strip()

                fname = dic["file_name"].strip()

                table_array = dic["table_array"]

                feat_array = dic["feature_array"]

                #annotations = dic["annotations"]

                block_types = dic["blocks"]

                if "embeddings" not in dic:
                    sheet = Sheet(np.array(table_array),
                            {"farr": feat_array, "name": table_id})
                else:
                    sheet = Sheet(np.array(table_array),
                            {"farr": feat_array, "name": table_id,
                             "embeddings": dic["embeddings"]})

                datatype_tags = np.empty(sheet.values.shape,
                                     dtype=CellTypePMF)

                if "data_types" in dic:
                    data_types = dic["data_types"]
                    #for i, row in enumerate(annotations):
                    for i, row in enumerate(data_types):
                        for j, cell in enumerate(row):
                            if cell is None:
                                typ = "empty"
                            else:
                                typ = cell
                            #elif cell == "derived":
                            #    typ = "data"
                            #elif cell == "notes":
                            #    typ = "metadata"

                            datatype_tags[i][j] = CellTypePMF({SemanticCellType.inverse_dict[typ]: 1})
                blks = []

                for (lx, ly, rx, ry, typ) in block_types:
                    new_blk = SimpleBlock(BlockTypePMF({FunctionBlockType.inverse_dict[typ]: 1.0}), ly, ry, lx, rx)
                    blks.append(new_blk)

                if len(blks) == 0:
                    continue

                self.tables.append(sheet)
                self.celltypes.append(datatype_tags)
                self.blocktypes.append(blks)

                if "layouts" in dic:
                    layout_types = dic["layouts"]

                    self.layouttypes.append(self.construct_layout_graph(blks, layout_types))
                else:
                    self.layouttypes.append(None)


    def split_tables(self, k=5, seed=0):

        temp = [i for i in range(len(self.tables))]

        random.seed(seed)

        random.shuffle(temp)

        indices = []

        each_len = int(len(temp) / k)

        for i in range(k):

            if i != k-1:

                indices.append(temp[i*each_len : (i+1)*each_len])
            else:
                indices.append(temp[i*each_len : ])

        return indices

    def split_indices(self, indices, k=5):
        new_indices = []

        each_len = int(len(indices) / k)

        for i in range(k):

            if i != k-1:

                new_indices.append(indices[i*each_len : (i+1)*each_len])
            else:
                new_indices.append(indices[i*each_len : ])

        return new_indices

    def get_table_from_index(self, index):
        return self.tables[index], self.celltypes[index], \
                self.blocktypes[index], self.layouttypes[index]

    def get_tables_from_indices(self, indices):

        sheet_list, celltype_list, blocktype_list, layouttype_list = [], [], [], []

        for k in indices:

            sheet_list.append(self.tables[k])

            celltype_list.append(self.celltypes[k])

            blocktype_list.append(self.blocktypes[k])

            layouttype_list.append(self.layouttypes[k])

        return sheet_list, celltype_list, blocktype_list, layouttype_list
