import pytest
import taxonomist.evaluate
import taxonomist.predictions
from taxonomist.taxonomist_model import TaxonomistModel
from taxonomist.input_config import TaxonomistModelArguments
from taxonomist.predictions import CombineCVPredictionsArgs, GroupPredictionsArgs
from taxonomist.evaluate import EvaluateArgs
from pathlib import Path
import taxonomist

def _find_ckpt(path):
    search_path = Path(path)
    ckpt_files = [x for x in search_path.glob("*.ckpt")]
    # Exclude files ending with "_last.ckpt"
    ckpt_files = [f for f in ckpt_files if not str(f).endswith("_last.ckpt")]

    # Get the first matching file, if available
    ckpt_path = str(ckpt_files[0]) if ckpt_files else None
    return ckpt_path

@pytest.mark.usefixtures("args_basic_train")

def test_opt_args():
    for opt_name in ['adam', 'adamw', 'sgd']:
        args = TaxonomistModelArguments(
            no_wandb=True,
            smoke_test=True,
            task="classification",
            dataset_config_path="conf/user_datasets.py",
            data_folder="data/raw/rodi/Induced_Organism_Drift_2022",
            dataset_name="rodi",
            csv_path="data/processed/rodi/01_rodi_processed_5splits_family.csv",
            label_column="family",
            fold=0,
            class_map_name="data/processed/rodi/rodi_label_map.txt",
            imsize=64,
            batch_size=4,
            aug='trivialaugment',
            load_to_memory=False,
            tta=False,
            model_name='mobilenetv3_small_075.lamb_in1k',
            opt=opt_name,
            opt_momentum=0.9,
            opt_weight_decay=0.0001,
            opt_beta1=0.9,
            opt_beta2=0.999,
            max_epochs=2,
            min_epochs=0,
            early_stopping=True,
            early_stopping_patience=10,
            criterion='cross-entropy',
            lr=0.0001,
            auto_lr=False,
            log_dir='roditest',
            out_folder='test_outputs',
            out_prefix='rodi',
            deterministic=True
        )
        tm = TaxonomistModel(args)
        tm.train()
