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
import time
#Process, Queue, current_process, freeze_support

def worker(func, input, output):

	for args in iter(input.get, 'STOP'):
		result = func(*args)
		#print(len(result[0]), result[1])
		output.put(result)


class BlockExtractorMCMC(BlockExtractor):

	def __init__(self, alpha=1.0, beta=0.1, lmd=10, num_process=50, N=50, maxdepth=None):
		# cius 0.03
		self.alpha = 1.0
		self.beta = beta
		self.lmd = lmd
		self.N = N
		self.num_process = num_process

	def is_split(self, d):

		p = self.alpha / ((1 + d) ** self.beta)

		rv = random.random()
		if rv <= p:
			return True
		else:
			return False

	def get_best_split_col(self, block, typ_array):
		best_split = None, None
		best_dist = 0
		dist_list = []
		entropy_list = []

		block_list = []
		lx, rx = block.get_top_row(), block.get_bottom_row()
		#print("current block! col split!", str(block))

		for b1, b2 in self.get_splits_cols(block):
			b1_ly, b1_ry = b1
			b2_ly, b2_ry = b2

			b1_np = self.get_mean_dist(typ_array[lx:rx+1, b1_ly:b1_ry + 1])
			b2_np = self.get_mean_dist(typ_array[lx:rx+1, b2_ly:b2_ry + 1])

			dist = np.linalg.norm(b1_np - b2_np)**2
			dist = 0 if dist == 0 else self.lmd * np.exp(-self.lmd / dist)
			dist_list.append(dist)
			block_list.append((b1, b2))
			#print("b1", b1_ly, b1_ry, "b2", b2_ly, b2_ry)
			#print("b1_np", b1_np, "b2_np", b2_np)
			#print("dist", np.linalg.norm(b1_np - b2_np)**2, "final", dist)

		sum_dist = sum(dist_list)

		if len(block_list) == 0 or sum_dist == 0:
			return (None, None)

		temp_list = [_ for _ in range(len(block_list))]

		rc = random.choices(population=temp_list,
							weights=dist_list)[0]
		b1 = SimpleBlock(None, block_list[rc][0][0], block_list[rc][0][1], lx, rx)
		b2 = SimpleBlock(None, block_list[rc][1][0], block_list[rc][1][1], lx, rx)
		#print("chosen", str(b1), str(b2))
		return (b1, b2)

	def get_entropy(self, ratios):
		ret = 0
		for r in ratios:
			if r == 0:
				continue
			ret -= r * np.log(r)
		return ret

	def get_mean_dist(self, vec):
		temp = vec.reshape(-1, vec.shape[-1])
		temp_sum = temp.sum(axis=0)
		if temp_sum.sum() == 0:
			return temp_sum
		else:
			return temp_sum / temp_sum.sum()

	def get_best_split(self, block, typ_array):

		best_split = None, None
		best_dist = 0
		dist_list = []
		entropy_list = []
		dist_c2v_list = []

		block_list = []
		#print("current block! row split!", str(block))

		for b1, b2 in self.get_splits_rows(block):

			(b1_lx, b1_rx) = b1
			(b2_lx, b2_rx) = b2

			b1_np = self.get_mean_dist(typ_array[b1_lx:b1_rx+1, :])
			b2_np = self.get_mean_dist(typ_array[b2_lx:b2_rx+1, :])

			dist = np.linalg.norm(b1_np - b2_np)**2
			dist = 0 if dist == 0 else self.lmd * np.exp(-self.lmd / dist)
			dist_list.append(dist)
			block_list.append((b1, b2))
			#print("b1", b1_lx, b1_rx, "b2", b2_lx, b2_rx)
			#print("b1_np", b1_np, "b2_np", b2_np)
			#print("dist", np.linalg.norm(b1_np - b2_np)**2, "final", dist)

		sum_dist = sum(dist_list)

		if len(block_list) == 0 or sum_dist == 0:
			return (None, None)

		temp_list = [_ for _ in range(len(block_list))]

		rc = random.choices(population=temp_list,
							weights=dist_list)[0]

		b1 = SimpleBlock(None, block.get_left_col(), block.get_right_col(), block_list[rc][0][0], block_list[rc][0][1])
		b2 = SimpleBlock(None, block.get_left_col(), block.get_right_col(), block_list[rc][1][0], block_list[rc][1][1])
		#print("chosen", str(b1), str(b2))
		return (b1, b2)

	def get_splits_cols(self, block):
		for col in range(block.get_left_col(), block.get_right_col()):
			b1 = (block.get_left_col(), col)
			b2 = (col + 1, block.get_right_col())

			yield b1, b2

	def get_splits_rows(self, block: SimpleBlock):

		for row in range(block.get_top_row(), block.get_bottom_row()):
			b1 = (block.get_top_row(), row)
			b2 = (row + 1, block.get_bottom_row())

			yield b1, b2

	def get_vec_each_cell(self, typ, dic):
		typ_obj = typ.get_best_type().str()

		#if typ_obj == "cardinal":
		#	temp_typ = "cardinal"
		if typ_obj in set(["ordinal", "cardinal"]):
			temp_typ = "cardinal"
		elif typ_obj == "nominal":
			temp_typ = "nominal"
		elif typ_obj == "datetime":
			temp_typ = "datetime"
		elif typ_obj == "empty":
			temp_typ = "empty"
		else:
			temp_typ = "string"

		vec = [0 for _ in range(len(dic))]

		vec[dic[temp_typ]] = 1

		return vec

	def convert_vec_for_all_cells(self, block, celltypes):
		lx, rx, ly, ry = block.get_top_row(), block.get_bottom_row(), block.get_left_col(), block.get_right_col()

		typ_dic = {}

		for (k, v) in SemanticCellType.inverse_dict.items():
			if k in set(["string", "cardinal", "datetime", "empty", "nominal"]):
				typ_dic[k] = len(typ_dic)
		#print("typ_dic", typ_dic)
		ret = []

		for i in range(lx, rx + 1):
			temp = []
			for j in range(ly, ry + 1):
				vec = self.get_vec_each_cell(celltypes[i][j], typ_dic)
				temp.append(vec)
			ret.append(temp)

		return np.array(ret)

	def convert_vec_for_nonemp_cells(self, block, celltypes):
		lx, rx, ly, ry = block.get_top_row(), block.get_bottom_row(), block.get_left_col(), block.get_right_col()

		typ_dic = {}

		for (k, v) in SemanticCellType.inverse_dict.items():
			if k in set(["string", "cardinal", "datetime", "empty", "nominal"]):
				typ_dic[k] = len(typ_dic)
		ret = []

		for i in range(lx, rx + 1):
			temp = []
			for j in range(ly, ry + 1):
				if celltypes[i][j].get_best_type().str() == "empty":
					vec = [0 for _ in range(len(typ_dic))]
				else:
					vec = self.get_vec_each_cell(celltypes[i][j], typ_dic)
				temp.append(vec)
			ret.append(temp)

		return np.array(ret)

	def generate_a_tree(self, sheet, start_block, typ_array, nonemp_array):
		row_blocks = []

		max_row, max_col = sheet.values.shape

		q = Queue()
		q.put((start_block, 0))

		weights = []

		pre_com = dict()

		while not q.empty():
			(next_block, d) = q.get()

			if self.is_split(d):
				(b1, b2) = self.get_best_split(next_block, typ_array)
				if (b1 and b2):
					q.put((b1, d+1))
					q.put((b2, d+1))
				else:
					row_blocks.append((next_block, d))
			else:

				row_blocks.append((next_block, d))

		blocks = set()

		q = Queue()
		for blk in row_blocks:
			q.put((blk[0], 0))

		while not q.empty():
			(next_block, d) = q.get()

			(b1, b2) = self.get_best_split_col(next_block, typ_array)

			if self.is_split(d):
				if (b1 and b2):
					q.put((b1, d+1))
					q.put((b2, d+1))
				else:
					blocks.add(next_block)
			else:
				blocks.add(next_block)

		weight = 0
		all_areas = sum([blk.get_area() for (blk, _) in row_blocks])
		for (blk, _) in row_blocks:
			arr = self.get_mean_dist(nonemp_array[blk.get_top_row():blk.get_bottom_row() + 1, blk.get_left_col():blk.get_right_col() + 1])
			#arr = self.get_cell_distribution_of_split(blk, tags, True)
			if arr is None:
				continue
			entropy = self.get_entropy(np.array(arr))
			weight += (blk.get_area() / all_areas) * entropy

		weight =  self.lmd * np.exp(-self.lmd * weight)
		return list(blocks), weight

	def extract_blocks(self, sheet: Sheet, tags, c2v_types) -> List[SimpleBlock]:
		#print(sheet.values[:, 0])

		random.seed(0)
		start = time.time()

		max_row, max_col = sheet.values.shape

		start_block = SimpleBlock(None, 0, max_col - 1, 0, max_row - 1)  # TODO: Check if -1 is correct

		typ_array = self.convert_vec_for_all_cells(start_block, tags)
		nonemp_array = self.convert_vec_for_nonemp_cells(start_block, tags)

		blocks_list, weight_list = [], []
		for i in range(self.N):
			(b, w) = self.generate_a_tree(sheet, start_block, typ_array, nonemp_array)
			blocks_list.append(b)
			weight_list.append(w)
		blocks = random.choices(population=blocks_list,
								weights=weight_list)[0]
		end = time.time()
		#print("choice running time", end - get_end)
		#print("main running time", end - start)
		return blocks
