import pytest

from taxonomist.taxonomist_model import TaxonomistModelArguments

def test_args():
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
        model_name='resnet18',
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

def test_no_checkpoint_but_resuming():
    with pytest.raises(ValueError) as excinfo:
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
            model_name='resnet18',
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
            deterministic=True,
            resume=True
        )
        assert("When resuming, a ckpt_path must be set" in str(excinfo.value))