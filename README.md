# Table Understanding System
This is the repository for the paper -- A Hybrid Probabilistic Approach for Table Understanding

## Input Format
The code requires the tables to be in json line format. Each json object is a single table containing the fields:

```
- table_array: a 2D list of cell contents
- table_id: a id (string)
- file_name: a file name (string)
- embeddings: a 3D list of cell vector representations (obtained from a pre-trained cell embedding model)
- feature_array: a 3D list of features
- blocks: a list such that each item represents a block (top index, left index, bottom index, right index, functional label)
- data_types: a 2D list of cell data types
- layouts: a list such that each item represents a relationship (block_a top index, block_a left index, block_a bottom index, block_a right index, block_b top index, block_b left index, block_b bottom index, block_b right index, relation type)
```

## Datasets
The processed datasets are available in `datasets.tar.gz`. It includes 4 available datasets. They do not contain the `embeddings` field. To get the cell representations, you can use the [pre-trained cell embedding model](https://github.com/majidghgol/TabularCellTypeClassification). The pre-processed datasets (`datasets_w_emb.tar.gz`) can be found [here](https://drive.google.com/drive/folders/1cPpZh1xqSivYc5YyPZ-npGv8DrXSrrfR?usp=sharing).

### The DG dataset
`dg_all.jl` has annotations of cell data types, blocks, and relationships between blocks.

### Other datasets
`cius_blocks.jl`, `saus_blocks.jl` and `deex_blocks.jl` have annotations for blocks. The blocks are automatically generated from cell-level labels. The original datasets can be found  [here](https://github.com/majidghgol/TabularCellTypeClassification/tree/master/annotations).

## Config File
See the example in `cfg/dg_config.yaml`.

## Generate Cross Validation Folds
```bash
python generate_folds.py --config cfg/dg_config.yaml
```

## Train Base Classifiers
```bash
python train_cl.py --config cfg/dg_config.yaml --cell --block
```

## Run the components
Run the cell classifier, the block detector and the layout predictor as follows. The predictions will be saved to the `results/dg/` directory. The output files are in json format including predictions for `k` folds. Each fold has a field `predict` presenting the predictions.
```
python test_cc.py --config cfg/dg_config.yaml --method psl
python test_be.py --config cfg/dg_config.yaml --method psl
python test_lp.py --config cfg/dg_config.yaml --method psl
```
