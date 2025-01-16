import pytest
import pandas as pd
from taxonomist.taxonomist_model import TaxonomistModelArguments

@pytest.fixture
def args_basic_train():
    args = TaxonomistModelArguments(
        no_wandb=True,
        task="classification",
        dataset_config_path="conf/user_datasets.py",
        data_folder="data/raw/rodi/Induced_Organism_Drift_2022",
        dataset_name="rodi",
        csv_path="data/processed/rodi/01_rodi_processed_5splits_family.csv",
        label_column="family",
        fold=0,
        class_map_name="data/processed/rodi/rodi_label_map.txt",
        imsize=224,
        batch_size=256,
        aug='trivialaugment',
        load_to_memory=False,
        tta=False,
        timm_model_name='resnet18',
        opt='adamw',
        max_epochs=2,
        min_epochs=0,
        early_stopping=True,
        early_stopping_patience=10,
        criterion='cross-entropy',
        lr=0.0001,
        auto_lr=False,
        log_dir='roditest',
        out_folder='outputs',
        out_prefix='rodi',
        deterministic=True
    )
    return args

@pytest.fixture
def args_basic_resume_train():
    args = TaxonomistModelArguments(
        no_wandb=True,
        task="classification",
        dataset_config_path="conf/user_datasets.py",
        data_folder="data/raw/rodi/Induced_Organism_Drift_2022",
        dataset_name="rodi",
        csv_path="data/processed/rodi/01_rodi_processed_5splits_family.csv",
        label_column="family",
        fold=0,
        class_map_name="data/processed/rodi/rodi_label_map.txt",
        imsize=224,
        batch_size=256,
        aug='trivialaugment',
        load_to_memory=False,
        tta=False,
        timm_model_name='resnet18',
        opt='adamw',
        max_epochs=5,
        min_epochs=0,
        early_stopping=True,
        early_stopping_patience=10,
        criterion='cross-entropy',
        lr=0.0001,
        auto_lr=False,
        log_dir='roditest',
        out_folder='outputs',
        out_prefix='rodi',
        deterministic=True,
        ckpt_path="tests/data/rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last.ckpt",
        resume=True
    )
    return args

@pytest.fixture
def args_basic_predict():
    args = TaxonomistModelArguments(
        no_wandb=True,
        task="classification",
        dataset_config_path="conf/user_datasets.py",
        data_folder="data/raw/rodi/Induced_Organism_Drift_2022",
        dataset_name="rodi",
        csv_path="data/processed/rodi/01_rodi_processed_5splits_family.csv",
        label_column="family",
        fold=0,
        class_map_name="data/processed/rodi/rodi_label_map.txt",
        imsize=224,
        batch_size=256,
        aug='none',
        tta=False,
        out_folder='outputs',
        out_prefix='',
        ckpt_path="tests/data/rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last.ckpt"
    )
    return args

@pytest.fixture
def args_basic_feature_extraction():
    args = TaxonomistModelArguments(
        no_wandb=True,
        task="feature-extraction",
        dataset_config_path="conf/user_datasets.py",
        data_folder="data/raw/rodi/Induced_Organism_Drift_2022",
        dataset_name="rodi",
        csv_path="data/processed/rodi/01_rodi_processed_5splits_family.csv",
        label_column="family",
        fold=0,
        class_map_name="data/processed/rodi/rodi_label_map.txt",
        imsize=224,
        batch_size=256,
        aug='none',
        tta=False,
        out_folder='outputs',
        out_prefix='',
        ckpt_path="tests/data/rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last.ckpt"
    )
    return args