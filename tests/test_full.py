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
def test_training_basic(args_basic_train):
    args = args_basic_train
    tm = TaxonomistModel(args)
    tm.train()

@pytest.mark.usefixtures("args_basic_train_regression")
def test_training_regression(args_basic_train_regression):
    args = args_basic_train_regression
    tm = TaxonomistModel(args)
    tm.train()

@pytest.mark.usefixtures("args_basic_train_with_mixup")
def test_training_with_mixup(args_basic_train_with_mixup):
    args = args_basic_train_with_mixup
    tm = TaxonomistModel(args)
    tm.train()

@pytest.mark.usefixtures("args_train_custom_model")
def test_training_custom_model(args_train_custom_model):
    args = args_train_custom_model
    tm = TaxonomistModel(args)
    tm.train()

@pytest.mark.usefixtures("args_basic_train_all_folds")
def test_training_all_folds(args_basic_train_all_folds):
    args = args_basic_train_all_folds
    tm = TaxonomistModel(args)
    tm.train()

def test_predict_all_folds():
    for fold in range(5):
        ckpt_path = _find_ckpt(f"test_outputs/rodi/rodi_allfolds_mobilenetv3_small_075.lamb_in1k/f{fold}/")
        args = TaxonomistModelArguments(
            task="classification",
            dataset_config_path="conf/user_datasets.py",
            data_folder="data/raw/rodi/Induced_Organism_Drift_2022",
            dataset_name="rodi",
            csv_path="data/processed/rodi/01_rodi_processed_5splits_family.csv",
            label_column="family",
            fold=fold,
            class_map_name="data/processed/rodi/rodi_label_map.txt",
            imsize=224,
            batch_size=512,
            aug='none',
            tta=False,
            out_folder='test_outputs',
            out_prefix='',
            ckpt_path=ckpt_path
        )
        tm = TaxonomistModel(args)
        tm.predict()

def test_predict_bioclip():
    ckpt_path = _find_ckpt(f"test_outputs/rodi/rodi_bioclip/f0/")
    args = TaxonomistModelArguments(
        task="classification",
        dataset_config_path="conf/user_datasets.py",
        data_folder="data/raw/rodi/Induced_Organism_Drift_2022",
        dataset_name="rodi",
        csv_path="data/processed/rodi/01_rodi_processed_5splits_family.csv",
        label_column="family",
        fold=0,
        class_map_name="data/processed/rodi/rodi_label_map.txt",
        imsize=224,
        batch_size=512,
        aug='none',
        tta=False,
        out_folder='test_outputs',
        out_prefix='',
        ckpt_path=ckpt_path
    )
    tm = TaxonomistModel(args)
    tm.predict()

def test_combine_cv_folds():
    args =  CombineCVPredictionsArgs(
        model_folder = "test_outputs/rodi/rodi_allfolds_mobilenetv3_small_075.lamb_in1k",
        tag = "rodi_none",
        reference_csv = "data/processed/rodi/01_rodi_processed_5splits_family.csv",
        reference_target = "family",
        suffix=".csv",
        n_folds = 5
    )
    taxonomist.predictions.combine_cv_predictions(args)

def test_group_predictions():
    args =  GroupPredictionsArgs(
    predictions = "test_outputs/rodi/rodi_allfolds_mobilenetv3_small_075.lamb_in1k/predictions/rodi_allfolds_mobilenetv3_small_075.lamb_in1k_rodi_none.csv",
    reference_csv = "data/processed/rodi/01_rodi_processed_5splits_family.csv",
    reference_target = "family",
    reference_group = "ind_id"
    )
    taxonomist.predictions.group_predictions(args)

def test_group_predictions_single_fold():
    args =  GroupPredictionsArgs(
    predictions = "test_outputs/rodi/rodi_allfolds_mobilenetv3_small_075.lamb_in1k/predictions/rodi_allfolds_mobilenetv3_small_075.lamb_in1k_rodi_none.csv",
    reference_csv = "data/processed/rodi/01_rodi_processed_5splits_family.csv",
    reference_target = "family",
    reference_group = "ind_id"
    )
    taxonomist.predictions.group_predictions(args)


def test_evaluate():
    args = EvaluateArgs(
    predictions = "test_outputs/rodi/rodi_allfolds_mobilenetv3_small_075.lamb_in1k/predictions/rodi_allfolds_mobilenetv3_small_075.lamb_in1k_rodi_none_grouped.csv",
    metric_config = "conf/eval.yaml",
    around=4
    )
    taxonomist.evaluate.evaluate(args)