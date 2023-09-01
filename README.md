# Taxonomist - a species classification pipeline

Taxonomist is a pipeline for classifying images of species, with a focus on scientific applications in natural sciences. It describes a simple framework that is easy to extend an modify for different needs. Taxonomist takes care of most parts of the classification pipeline (training, cross-validation, logging, evaluation) and lets you focus on designing the experiments and analyzing the results of different classification approaches. 

Features:
- Image classification and regression with state-of-the-art Deep Learning models from the [PyTorch Image Models (`timm`)](https://timm.fast.ai/) library.
- Transparent and easy to modify. Operates around simple python scripts and `.csv`-files without opaque modules and functions with side-effects.
- Opinionated folder structure designed for scientific, reproducible experiments.
- Easy result comparisons between experiments and across datasets.
- Produces results in commonly used `.csv` format that can be further analyzed with other tools
- Implements best practices for classifier evaluation, such as bootstrap confidence intervals and cross-validation.

In essence, Taxonomist is a framework around [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), providing an opinionated project structure for scientific experiments using supervised learning on hierarchical data.
# Installation

Clone the repository

```bash
git clone https://github.com/mikkoim/taxonomist.git
cd taxonomist
```

Install anaconda or miniconda, for example by running the commands:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

For windows, see the installation instructions for Miniconda.

Installing [Mamba](https://github.com/mamba-org/mamba) is recommended as it makes the installation much faster than with conda. In the ```(base)``` environment, run
```bash
conda install mamba -n base -c conda-forge
```
Now you can replace the ```conda``` commands with ```mamba``` when installing packages.

Next, Create the environment

```bash
mamba env create -f environment.yml
conda activate taxonomist
```

And install the library package:

```bash
pip install -e .
```

# Overview

To use Taxonomist with your own data, you have to produce data loading functions to make your dataset compatible with the pipeline.

1. Load datasets
    - Write loading instructions (examples in `docs/dataset_loading.md`)
    - (Optional: Perform raw data analysis (examples in `notebooks/`))
2. Preprocessing
    - Create a preprocessing script (examples in `scripts/preprocessing/`)
    - Add data loading functions to the library (examples in `src/taxonomist/datasets.py`)

3. Workflow document
    - Document your workflow (commands etc.) into a workflow file. (Examples in `docs/workflows`)

When these steps are complete, Taxonomist can automate the rest of the classification pipeline:

4. Train-test-val -splits
5. Training
6. Prediction
    - Prediction with test-time augmentation
    - Prediction for other datasets 
7. Evaluation
    - Cross-validation
    - Grouping
8. Comparison

Each step produces intermediary files in csv format, making custom analysis and modifications easy.

# File system
Taxonomist is based on an opinionated file system that produces following output files:

- Model checkpoints (weights, hyperparameters)
- Train-time augmentation visualizations
- Predict-time augmentation visualizations
- Prediction outputs as csv, with softmax scores for all classes
- Grouped prediction outputs
- Metrics for several models in csv format

```
Outputs
├── Dataset A
│   ├── Model A1
│   │   ├── fold 0
│   │   │   ├── model_a1-f0.ckpt (model weights)
│   │   │   ├── aug-model_a1(augmentation visualizations)
│   │   │   │   ├── aug-test.png
│   │   │   │   ├── aug-train.png
│   │   │   │   └── aug-val.png
│   │   │   └── predictions (prediction outputs)
│   │   │       ├── metrics
│   │   │       │   └── model_a1-f0-test_aug_preds_metrics.csv
│   │   │       ├── test_aug1
│   │   │       │   ├── model_a1-f0-test_aug1_preds.csv
│   │   │       │   └── model_a1-f0-test_aug1_preds_grouped.csv
│   │   │       └── test_aug2
│   │   │           └── ...
│   │   └── fold 1
│   │       └── ...
│   └── Model A2
│       └── ...
├── Dataset B
│   ├── Model B1
│   └── ...
└── ...
```