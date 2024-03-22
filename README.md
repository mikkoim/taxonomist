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

Note that Taxonomist is still under heavy development and large changes can be introduced!

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

# Getting started

The workflow in [docs/workflows/00_workflow_rodi.md](docs/workflows/00_workflow_rodi.md) is a good place to start. It walks through all the features, and the model training takes around 30 minutes with a decent GPU. 

# Overview

To use Taxonomist with your own data, you have to produce data loading functions to make your dataset compatible with the pipeline.

1. **Load and get to know your dataset**

    Analyze your dataset structure and find a way to represent the dataset in a table format. The table should contain the at least the following columns:

    - filename
    - label

    If the dataset has a nested folder structure, columns that specify the location in the folder structure are needed. Also, if there is a grouping among the images, for example if there are several images from a specimen, a grouping identifier is needed.

    | label | folder | individual | filename |
    | --- | --- | --- | --- |
    | cat | felines | A | 01.png |
    | cat | felines | A | 02.png |
    | dog | canines | B | 03.png |
    | dog | canines | B | 04.png |
    | wolf | canines | C | 05.png |

    The columns can also contain any other metadata that is seemed useful.

    It is useful to create a separate `data` folder, with two subfolders:
    - `raw`: Contains raw data that should be immutable
    - `processed`: Contains files that are processed from the raw data using scripts, like the preprocessing scripts.

1. **Preprocessing**
    - Create a preprocessing script that reads the filenames in your dataset and produces a table like above (examples in `scripts/preprocessing/`). The preprocessing script should also create a list of all the labels in the dataset into a text file. This label list is used as a label mapping, ensuring that all labels get a proper index even for folds where all labels are not present.
    - Add data loading functions to the library (examples in `src/taxonomist/datasets.py`)
    
    The data loading function should be able resolve into a full path when given a root directory. The root directory can be specified during training so the data location can change without changing the dataset table.


When these steps are complete, Taxonomist can automate the rest of the classification pipeline:

3. **Train-test-val -splits**: Handles splitting the dataset into train, test and validation splits, handling stratification and possible groups where data leakage could occur. The test sets are mutually exclusive, together becoming the full dataset.
3. **Training**: Trains a deep neural network of your choice from the architectures supported by `timm`. 
3. **Prediction**: 
    - Prediction with test-time augmentation
    - Prediction can be done easily across datasets, with previously trained models and mixed label sets.
3. **Evaluation**:
    - Cross-validation
    - Grouping: If the unit of classification is a group, such as an individual specimen, all classifications from this group can be aggregated to produce better estimates.
3. **Comparison**: Comparison between models, datasets and experiments is easy with the flexible comparison script. It produces a csv file with all the results for easy analysis. Experiments, models and datasets can be tagged with arbitary tags that appear in the final result table.

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