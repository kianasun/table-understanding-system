from cell_classifier.cell_classifier import CellClassifier
import pickle
import numpy as np
from type.cell.function_cell_type import FunctionCellType
from type.cell.cell_type_pmf import CellTypePMF
from reader.sheet import Sheet
from typing import List
import sys
import torch
import numpy as np


class C2VCellClassifier(CellClassifier):
    def __init__(self, cl_model_path, config):
        self.config = config
        infersent_source = config["c2v"]['infersent_source']
        sys.path.append(infersent_source)

        sys.path.append(config["c2v"]['ce_source'])
        from models import ClassificationModel, CEModel, FeatEnc
        from excel_toolkit import get_sheet_names, get_sheet_tarr, get_feature_array
        from helpers import Preprocess, SentEnc, label2ind
        from test_cl import predict_labels, predict_with_embs
        print(cl_model_path)

        ce_model_path = config["c2v"]['ce_model']
        #cl_model_path = config['cl_model']
        fe_model_path = config["c2v"]['fe_model']
        self.w2v_path = config["c2v"]['w2v']
        self.vocab_size = config["c2v"]['vocab_size']
        self.infersent_model = config["c2v"]['infersent_model']

        self.mode = 'ce+f'
        self.device = 'cpu'
        ce_dim = 512
        senc_dim = 4096
        window = 2
        f_dim = 43
        fenc_dim = 40
        n_classes = 4
        if self.device != 'cpu': torch.cuda.set_device(self.device)

        self.ce_model = CEModel(senc_dim, ce_dim//2, window*4)
        self.ce_model = self.ce_model.to(self.device)
        self.fe_model = FeatEnc(f_dim, fenc_dim)
        self.fe_model = self.fe_model.to(self.device)
        self.cl_model = ClassificationModel(ce_dim+fenc_dim, n_classes).to(self.device)

        self.ce_model.load_state_dict(torch.load(ce_model_path, map_location=self.device))
        self.fe_model.load_state_dict(torch.load(fe_model_path, map_location=self.device))
        self.cl_model.load_state_dict(torch.load(cl_model_path, map_location=self.device))

        self.label2ind = ["attributes", "data", "header", "metadata"]

    def generate_sent(self, tarr):
        sentences = set()
        for row in tarr:
            for c in row:
                sentences.add(c)
        return sentences

    def get_result_from_embs(self, sheet):

        sys.path.append(self.config["c2v"]['ce_source'])
        from test_cl import predict_labels, predict_with_embs

        labels, probs = predict_with_embs(sheet.meta["embeddings"], self.cl_model)

        probs = np.exp(probs)

        labels = np.vectorize(lambda x: self.label2ind[x])(labels)

        result = dict(text=sheet.meta["farr"],
                      labels=labels,
                      labels_probs=probs.tolist())

        return result

    def get_result(self, table, senc, tarr):

        labels, probs, features = predict_labels(table, self.cl_model,
                                       self.ce_model,
                                       self.fe_model,
                                       senc, self.mode,
                                       self.device)

        probs = np.exp(probs)

        labels = np.vectorize(lambda x: self.label2ind[x])(labels)

        result = dict(text=tarr.tolist(), labels=labels.tolist(),
                      labels_probs=probs.tolist())

        return result, features

    def generate_c2v_from_array(self, tarr, tfeat):
        senc = SentEnc(self.infersent_model, self.w2v_path,
                       self.vocab_size, device=self.device, hp=False)
        prep = Preprocess()

        table = dict(table_array=tarr, feature_array=tfeat)

        sentences = self.generate_sent(tarr)

        senc.cache_sentences(list(sentences))

        return self.get_result(table, senc, tarr)

    def generate_c2v_from_file(fname, sname):

        senc = SentEnc(self.infersent_model, self.w2v_path,
                       self.vocab_size, device=self.device, hp=False)
        prep = Preprocess()

        tarr, n, m = get_sheet_tarr(fname, sname, file_type='xls')

        ftarr = get_feature_array(fname, sname, file_type='xls')

        table = dict(table_array=tarr, feature_array=ftarr)

        sentences = self.generate_sent(tarr)

        senc.cache_sentences(list(sentences))

        return self.get_result(table, senc, tarr)

    def __predict_wrapper(self, res, r, c):

        pred = np.empty((r, c), dtype=CellTypePMF)

        labels = res['labels']
        probs = res['labels_probs']

        for i in range(r):
            for j in range(c):
                t_l = labels[i][j]
                """
                if t_l == "notes":
                    t_l = "metadata"
                elif t_l == "derived":
                    t_l = "data"
                """
                cell_class_dict = {
                    FunctionCellType.inverse_dict[t_l]: probs[i][j]
                }
                pred[i][j] = CellTypePMF(cell_class_dict)

        return pred

    def generate_labels_from_a_sheet(self, sheet: Sheet):
        result = None

        if "farr" in sheet.meta:
            result = self.generate_c2v_from_array(sheet.values, sheet.meta['farr'])
        else:
            print("has path")
            path = sheet.meta['path']

            if path.endswith("xls") or path.endswith("xlsx"):
                file_path = path
                sheet_name = sheet.meta['name']
            else:
                file_path = ".".join(path.split(".")[:-1] + ["temp", "xls"])
                values = [[cell[:32767] for cell in row] for row in sheet.values]
                pyx.save_as(array=values, dest_file_name=file_path)
                sheet_name = "pyexcel_sheet1"

            result = self.generate_c2v_from_file(file_path, sheet_name)
        return result

    def classify_cells(self, sheet: Sheet) -> 'np.ndarray[CellTypePMF]':

        #(result, _) = self.generate_labels_from_a_sheet(sheet)

        result = self.get_result_from_embs(sheet)

        nr, nc = sheet.values.shape

        tags = self.__predict_wrapper(result, nr, nc)

        return tags

    def classify_cells_all_tables(self, sheets):
        tags = []

        for sheet in sheets:

            tags.append(self.classify_cells(sheet))

        return tags

    def generate_features(self, sheet: Sheet):

        (_, features) = self.generate_labels_from_a_sheet(sheet)

        return features
