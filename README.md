# Taxonomist - a species classification pipeline

Taxonomist is a pipeline for classifying images of species, with a focus on scientific applications in natural sciences. It describes a simple framework that is easy to extend an modify for different needs. Taxonomist takes care of most parts of the classification pipeline (training, cross-validation, logging, evaluation) and lets you focus on designing the experiments and analyzing the results of different classification approaches. 

Features:
- Image classification and regression with state-of-the-art Deep Learning models from the [PyTorch Image Models (`timm`)](https://timm.fast.ai/) library.
- Easy to modify. Operates around simple python scripts and `.csv`-files without complicated modules
- Opinionated folder structure designed for scientific, reproducible experiments
- Easy result comparisons between experiments
- Produces results in commonly used `.csv` format that can be further analyzed with other tools
- Implements best practices for classifier evaluation

In essence, Taxonomist is a framework around [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) which itself is a framework for the PyTorch deep learning library. 
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
    - Write loading instructions (example in `docs/dataset_loading.md`)
    - (Optional: Perform raw data analysis (example in `notebooks/01_raw_data_analysis_rodi.ipynb`))
2. Preprocessing
    - Create a preprocessing script (example in `scripts/preprocessing/process_rodi.py`)
    - Add data loading functions to the library (example in `src/taxonomist/datasets.py`)

3. Workflow document
    - Document your workflow (commands etc.) into a workflow file. (Example in `docs/workflows/00_workflow_rodi.md`)

When these steps are complete, Taxonomist can automate the rest of the classification pipeline:

4. Training
5. Prediction
6. Evaluation
    - Cross-validation
    - Grouping
7. Comparison

The final product is a `.csv` of cross-validated results with your dataset, that you can analyze and visualize with tools of your choice. (Examples coming soon.)
