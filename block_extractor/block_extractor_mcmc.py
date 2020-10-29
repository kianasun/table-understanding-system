import numpy as np
from block_extractor.block_extractor import BlockExtractor
from block_extractor.block_extractor_v2 import BlockExtractorV2
from type.block.simple_block import SimpleBlock
from typing import List
from queue import Queue
from reader.sheet import Sheet
from type.cell.cell_type_pmf import CellTypePMF
from type.cell.cell_type import CellType
from typing import Tuple
import random
from sklearn.preprocessing import normalize
from type.cell.semantic_cell_type import SemanticCellType
import multiprocessing as mp
from block_extractor.block_extractor_c2v import BlockExtractorC2V
from type.cell.function_cell_type import FunctionCellType
from type.cell.cell_type_pmf import CellTypePMF
from type.block.function_block_type import FunctionBlockType
#Process, Queue, current_process, freeze_support

def worker(func, input, output):

    for args in iter(input.get, 'STOP'):
        result = func(*args)
        #print(len(result[0]), result[1])
        output.put(result)


class BlockExtractorMCMC(BlockExtractor):

    def __init__(self, beta=0.1, lmd=10, num_process=50, N=50):
        # cius 0.03
        self.alpha = 1.0
        self.beta = beta
        self.lmd = lmd
        self.N = N
        self.num_process = num_process

    def get_hypotheses(self, blocks):
        row_h = set()
        col_h = set()

        for block in blocks:
            row_h.add(block.get_top_row() - 1)
            row_h.add(block.get_bottom_row())
            col_h.add(block.get_left_col() - 1)
            col_h.add(block.get_right_col())

        return row_h, col_h

    def get_cell_dist_of_split_c2v(self, block, c2vtypes, tags):
        block_dist = {v.id(): 0 for (k, v) in FunctionCellType.inverse_dict.items()}
        lx, ly, rx, ry = block.top_row, block.left_col, block.bottom_row, block.right_col

        for i in range(lx, rx + 1):
            for j in range(ly, ry + 1):
                if tags[i][j].get_best_type().str() == "empty":
                    continue

                typ = c2vtypes[i][j].get_best_type().str()

                temp_typ = FunctionCellType.inverse_dict[typ].id()

                block_dist[temp_typ] += 1

        block_area = block.get_area()

        if block_area == 0:
            return None

        return [block_dist[i] / block_area for i in block_dist]

    def get_block_size(self, block):
        block_size = (block.get_right_col() - block.get_left_col() + 1) * \
                     (block.get_bottom_row() - block.get_top_row() + 1)

        return block_size

    def is_split(self, block, d, ratio):
        #total_size = self.get_block_size(block)

        #ratio = total_size / area

        p = self.alpha / ((1 + d) ** self.beta)
        #p = p * ratio
        #p = self.alpha / (1 + self.beta * d)


        #print("orig p", p)
        #print(block)
        #p *= ratio

        rv = random.random()

        #print("is_split, p, ", p, "rv, ", rv)
        if rv <= p:
            #print("split")
            return True
        else:
            return False

    def get_best_split_col(self, block, col_h, celltypes):
        r, c = celltypes.shape

        best_split = None, None
        best_dist = 0
        dist_list = []
        entropy_list = []

        block_list = []

        for b1, b2 in self.get_splits(block, set(), col_h):

            b1_np = np.array(self.get_cell_distribution_of_split(b1, celltypes))
            b2_np = np.array(self.get_cell_distribution_of_split(b2, celltypes))

            dist = np.linalg.norm(b1_np - b2_np)**2
            dist = 0 if dist == 0 else self.lmd * np.exp(-self.lmd / dist)
            dist_list.append(dist)
            block_list.append((b1, b2))

        sum_dist = sum(dist_list)

        if len(block_list) == 0 or sum_dist == 0:
            return (None, None), 0

        temp_list = [_ for _ in range(len(block_list))]

        rc = random.choices(population=temp_list,
                            weights=dist_list)[0]
        return block_list[rc], dist_list[rc]

    def get_cell_distribution_of_split(self, block, celltypes, ignore=False):
        block_dist = {v.id(): 0 for (k, v) in SemanticCellType.inverse_dict.items()
                        if k in set(["string", "cardinal", "datetime", "empty"])}
        #block_dist = {v.id(): 0 for (k, v) in SemanticCellType.inverse_dict.items()
        #              }

        lx, ly, rx, ry = block.top_row, block.left_col, block.bottom_row, block.right_col

        for i in range(lx, rx + 1):
            for j in range(ly, ry + 1):
                typ = celltypes[i][j].get_best_type().str()
                temp_typ = None
                if typ in set(["cardinal", "ordinal", "nominal"]):
                    temp_typ = "cardinal"
                elif typ == "datetime":
                    temp_typ = "datetime"
                elif typ == "empty":
                    temp_typ = "empty"
                else:
                    temp_typ = "string"
                #temp_typ = typ

                temp_typ = SemanticCellType.inverse_dict[temp_typ].id()

                #if typ not in block_dist:
                #    block_dist[typ] = 0

                block_dist[temp_typ] += 1

        block_area = block.get_area()

        if ignore:
            empty_id = SemanticCellType.inverse_dict["empty"].id()
            block_area -= block_dist[empty_id]
            block_dist[empty_id] = 0

        if block_area == 0:
            return None

        return [block_dist[i] / block_area for i in block_dist]

    def get_entropy(self, ratios):
        ret = 0
        for r in ratios:
            if r == 0:
                continue
            ret -= r * np.log(r)
        return ret

    def get_best_split(self, block, row_h, col_h, celltypes, row_dist, row_c2v_dist, pre_com):
        r, c = celltypes.shape

        best_split = None, None
        best_dist = 0
        dist_list = []
        entropy_list = []
        dist_c2v_list = []

        block_list = []

        for b1, b2 in self.get_splits_rows(block, row_h):

            #b1_lx, b1_rx = b1.top_row, b1.bottom_row
            #b2_lx, b2_rx = b2.top_row, b2.bottom_row
            (b1_lx, b1_rx) = b1
            (b2_lx, b2_rx) = b2

            if (b1_lx, b1_rx) not in pre_com:
                b1_np = np.mean(row_dist[b1_lx:b1_rx+1, :], axis=0)
                pre_com[(b1_lx, b1_rx)] = b1_np
            else:
                b1_np = pre_com[(b1_lx, b1_rx)]

            if (b2_lx, b2_rx) not in pre_com:
                b2_np = np.mean(row_dist[b2_lx:b2_rx+1, :], axis=0)
                pre_com[(b2_lx, b2_rx)] = b2_np
            else:
                b2_np = pre_com[(b2_lx, b2_rx)]

            #b1_c2v_np = np.mean(row_c2v_dist[b1_lx:b1_rx+1, :], axis=0)

            #b2_c2v_np = np.mean(row_c2v_dist[b2_lx:b2_rx+1, :], axis=0)

            dist = np.linalg.norm(b1_np - b2_np)**2
            #dist_c2v = np.linalg.norm(b1_c2v_np - b2_c2v_np)**2
            #dist = (dist + dist_c2v) / 2
            #dist_c2v_list.append(dist_c2v)
            dist = 0 if dist == 0 else self.lmd * np.exp(-self.lmd / dist)
            dist_list.append(dist)
            #dist_list.append(dist)
            block_list.append((b1, b2))

        sum_dist = sum(dist_list)

        if len(block_list) == 0 or sum_dist == 0:
            return (None, None), 0

        #print("prev", dist_list)
        #dist_list = normalize([[_] for _ in dist_list], axis=0)
        #dist_list = [_[0] for _ in dist_list]

        #print("after", dist_list)
        #dist_list = [(_/sum_dist) * _ for _ in dist_list]

        temp_list = [_ for _ in range(len(block_list))]

        rc = random.choices(population=temp_list,
                            weights=dist_list)[0]
        #print("selected", dist_list[rc], (str(block_list[rc][0]), str(block_list[rc][1])))
        #print("dist_list", dist_list)
        #print("dist_c2v", dist_c2v_list)
        #print("entroy_list", entropy_list)
        #print("block_list", [(str(_[0]), str(_[1])) for _ in block_list])

        b1 = SimpleBlock(None, block.get_left_col(), block.get_right_col(), block_list[rc][0][0], block_list[rc][0][1])
        b2 = SimpleBlock(None, block.get_left_col(), block.get_right_col(), block_list[rc][1][0], block_list[rc][1][1])
        return (b1, b2), dist_list[rc]

    def get_splits(self, block: SimpleBlock, row_h, col_h):

        for row in row_h:
            if block.get_top_row() <= row < block.get_bottom_row():
                b1 = SimpleBlock(None, block.get_left_col(), block.get_right_col(), block.get_top_row(), row)
                b2 = SimpleBlock(None, block.get_left_col(), block.get_right_col(), row + 1, block.get_bottom_row())

                yield b1, b2

        for col in col_h:
            if block.get_left_col() <= col < block.get_right_col():
                b1 = SimpleBlock(None, block.get_left_col(), col, block.get_top_row(), block.get_bottom_row())
                b2 = SimpleBlock(None, col + 1, block.get_right_col(), block.get_top_row(), block.get_bottom_row())

                yield b1, b2

    def get_splits_rows(self, block: SimpleBlock, row_h):

        for row in row_h:
            if block.get_top_row() <= row < block.get_bottom_row():
                b1 = (block.get_top_row(), row)
                b2 = (row + 1, block.get_bottom_row())

                #b1 = SimpleBlock(None, block.get_left_col(), block.get_right_col(), block.get_top_row(), row)
                #b2 = SimpleBlock(None, block.get_left_col(), block.get_right_col(), row + 1, block.get_bottom_row())

                yield b1, b2

    def get_dist_for_all_rows(self, block, celltypes):
        lx, rx = block.get_top_row(), block.get_bottom_row()
        ret = []

        for i in range(lx, rx + 1):

            temp_block = SimpleBlock(None, block.get_left_col(), block.get_right_col(), i, i)

            arr = self.get_cell_distribution_of_split(temp_block, celltypes)

            ret.append(arr)

        return np.array(ret)

    def get_dist_for_all_rows_c2v(self, block, c2vtypes, tags):
        lx, rx = block.get_top_row(), block.get_bottom_row()
        ret = []

        for i in range(lx, rx + 1):

            temp_block = SimpleBlock(None, block.get_left_col(), block.get_right_col(), i, i)

            arr = self.get_cell_dist_of_split_c2v(temp_block, c2vtypes, tags)

            ret.append(arr)

        return np.array(ret)

    def generate_a_tree(self, sheet, row_h, col_h, start_block, tags, row_dist, row_c2v_dist):
        row_blocks = []

        max_row, max_col = sheet.values.shape

        #print("within")
        q = Queue()
        q.put((start_block, 0))
        #row_q = Queue()

        weights = []

        pre_com = dict()

        import time
        start = time.time()
        while not q.empty():
            (next_block, d) = q.get()
            #print(str(next_block), d)
            #print(b1, b2, split_w)


            #print(str(next_block))
            if self.is_split(next_block, d, (next_block.bottom_row - next_block.top_row + 1) / max_row):
                (b1, b2), split_w = self.get_best_split(next_block, row_h, set(), tags, row_dist, row_c2v_dist, pre_com)
                if (b1 and b2):
                    #print("split", str(b1), str(b2))
                    q.put((b1, d+1))
                    q.put((b2, d+1))
                    #weights.append(split_w)
                else:
                    row_blocks.append((next_block, d))
                    #row_q.put((next_block, d))
                    #row_blocks.add((next_block, d))
                #print(b1, b2)
            else:

                row_blocks.append((next_block, d))
                #row_q.put((next_block, d))
                #row_blocks.add((next_block, d))

        mid = time.time()
        blocks = set()

        q = Queue()
        for blk in row_blocks:
            q.put((blk[0], blk[1]))
        #self.beta = 0.

        #print("col")

        while not q.empty():
            (next_block, d) = q.get()

            (b1, b2), split_w = self.get_best_split_col(next_block, col_h, tags)

            if self.is_split(next_block, d, max_row * max_col):
                if (b1 and b2):
                    q.put((b1, d+1))
                    q.put((b2, d+1))
                else:
                    #blocks.append(next_block)
                    blocks.add(next_block)
            else:
                #blocks.append(next_block)
                blocks.add(next_block)

        end = time.time()
        #print("a tree: {}, {}, {}".format(mid - start, end - mid, end - start))
        #print(len(blocks), len(weights))
        weight = 0
        all_areas = sum([blk.get_area() for (blk, _) in row_blocks])
        for (blk, _) in row_blocks:
            arr = self.get_cell_distribution_of_split(blk, tags, True)
            if arr is None:
                continue
            entropy = self.get_entropy(np.array(arr))
            weight += (blk.get_area() / all_areas) * entropy

        weight =  self.lmd * np.exp(-self.lmd * weight)
        #weight = 100 if weight == 0 else np.exp(1 / weight)
        return list(blocks), weight

    def convert_c2v_cell(self, c2v_types):
        (r, c) = c2v_types.shape
        pred = np.empty((r, c), dtype=CellTypePMF)

        for i in range(r):
            for j in range(c):

                cell_class_dict = {
                    FunctionCellType.inverse_dict[c2v_types[i][j].get_best_type().str()]: 1.0
                }
                pred[i][j] = CellTypePMF(cell_class_dict)

        return pred

    def convert_basic(self, types):
        (r, c) = types.shape
        pred = np.empty((r, c), dtype=CellTypePMF)

        for i in range(r):
            for j in range(c):

                typ = types[i][j].get_best_type().str()

                if typ in ["nominal", "ordinal"]:
                    typ = "cardinal"
                elif typ in ["person", "organization", "location", "event"]:
                    typ = "string"

                cell_class_dict = {
                    SemanticCellType.inverse_dict[typ]: 1.0
                }
                pred[i][j] = CellTypePMF(cell_class_dict)

        return pred

    def extract_blocks(self, sheet: Sheet, tags, c2v_types) -> List[SimpleBlock]:

        random.seed(2020)

        bev = BlockExtractorV2()

        c2v_tags = self.convert_c2v_cell(c2v_types)

        temp_tags = self.convert_basic(tags)

        row_blocks = bev.merge_sheet_left_to_right(sheet, temp_tags)

        maximal_blocks = bev.merge_sheet_top_to_bottom(row_blocks)

        cand_blocks = [_ for t, _ in maximal_blocks]

        #cand_blocks = bev.extract_blocks(sheet, c2v_tags)

        #row_h, col_h = self.get_hypotheses(cand_blocks)

        max_row, max_col = sheet.values.shape

        #print("r, c", max_row, max_col)
        #print("row_h", row_h)

        row_h = set([_ for _ in range(0, max_row)])

        col_h = set([_ for _ in range(0, max_col)])

        start_block = SimpleBlock(None, 0, max_col - 1, 0, max_row - 1)  # TODO: Check if -1 is correct

        row_dist = self.get_dist_for_all_rows(start_block, tags)

        #row_c2v_dist = self.get_dist_for_all_rows_c2v(start_block, c2v_tags, tags)
        row_c2v_dist = []

        #freeze_support()
        blocks_list, weight_list = [], []

        NUMBER_OF_PROCESSES = self.num_process
        TASKS = [[sheet, row_h, col_h, start_block, tags, row_dist, row_c2v_dist]
                  for i in range(self.N)]

        # Create queues
        task_queue = mp.Queue()
        done_queue = mp.Queue()

        for task in TASKS:
            task_queue.put(task)

        # Start worker processes
        for i in range(NUMBER_OF_PROCESSES):
            mp.Process(target=worker, args=(self.generate_a_tree,
                                        task_queue, done_queue)).start()

        # Get and print results
        for i in range(len(TASKS)):
            blks, ws = done_queue.get()
            """
            print(i, ws)
            for blk in blks:
                print(blk)
            print()
            """
            blocks_list.append(blks)
            weight_list.append(ws)

        #print("block len", len(blocks_list))
        for i in range(NUMBER_OF_PROCESSES):
            task_queue.put('STOP')

        blocks = random.choices(population=blocks_list,
                                weights=weight_list)[0]
        """
        print("selected")
        for blk in blocks:
            print(blk)
        """
        #print("len blocks", len(blocks))

        return blocks
