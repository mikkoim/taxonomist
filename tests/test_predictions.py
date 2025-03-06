import pytest
from taxonomist.taxonomist_model import TaxonomistModel
import pandas as pd
from taxonomist.input_config import TaxonomistModelArguments
import pandas as pd


@pytest.mark.usefixtures("args_basic_predict")
def test_predict_basic(args_basic_predict):
    args = args_basic_predict
    tm = TaxonomistModel(args)
    tm.predict()
    df = pd.read_csv(tm.path_manager.predict_fpath)
    assert 'Salmonidae' in df.columns
    assert df.shape == (1014, 10)

def test_predict_with_parquet():
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
        out_folder='test_outputs',
        out_prefix='',
        ckpt_path="tests/data/rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last.ckpt",
        prediction_format="parquet"
    )
    tm = TaxonomistModel(args)
    tm.predict()
    df = pd.read_parquet(tm.path_manager.predict_fpath)
    assert 'Salmonidae' in df.columns
    assert df.shape == (1014, 9)

def test_predict_softmax():
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
        out_folder='test_outputs',
        out_prefix='',
        ckpt_path="tests/data/rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last.ckpt",
        prediction_format="csv",
        return_softmax=True
    )
    tm = TaxonomistModel(args)
    tm.predict()
    df = pd.read_csv(tm.path_manager.predict_fpath)
    assert df.iloc[:,3:].sum(axis=1).all() == pytest.approx(1)
    assert df.shape == (1014, 10)

def test_predict_with_train_and_val():
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
        out_folder='test_outputs',
        out_prefix='trainvaltest',
        ckpt_path="tests/data/rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last.ckpt",
        prediction_format="parquet",
        predict_on_val=True,
        predict_on_train=True
    )
    tm = TaxonomistModel(args)
    tm.predict()
    ref = pd.read_csv(args.csv_path)
    df_test = pd.read_parquet(tm.path_manager.predict_fpath)
    df_train = pd.read_parquet(tm.path_manager.predict_fpath_with_prefix("train"))
    df_val = pd.read_parquet(tm.path_manager.predict_fpath_with_prefix("val")) 
    assert 'Salmonidae' in df_test.columns
    assert 'Salmonidae' in df_train.columns
    assert 'Salmonidae' in df_val.columns
    assert df_test.shape[0] == ref.query("`0` == 'test'").shape[0]
    assert df_train.shape[0] == ref.query("`0` == 'train'").shape[0]
    assert df_val.shape[0] == ref.query("`0` == 'val'").shape[0]

@pytest.mark.usefixtures("args_basic_feature_extraction")
def test_feature_extraction(args_basic_feature_extraction):
    args = args_basic_feature_extraction
    tm = TaxonomistModel(args)
    tm.predict()
    features = pd.read_parquet("tests/data/features/rodi_none/rodi_resnet18_f0_250113-1038-a44d_pooled.parquet.gzip")
    ref = pd.read_csv(args.csv_path).query("`0` == 'test'")
    assert len(features) == len(ref)


@pytest.mark.usefixtures("args_feature_extraction_pretrained")
def test_feature_extraction_pretrained(args_feature_extraction_pretrained):
    args = args_feature_extraction_pretrained
    tm = TaxonomistModel(args)
    tm.predict()
    features = pd.read_parquet(tm.path_manager.predict_fpath)
    ref = pd.read_csv(args.csv_path).query("`0` == 'test'")
    assert len(features) == len(ref)

def test_feature_extraction_with_train_and_val():
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
        out_folder='test_outputs',
        out_prefix='',
        ckpt_path="tests/data/rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last.ckpt",
        feature_extraction='pooled',
        predict_on_val=True,
        predict_on_train=True
    )
    tm = TaxonomistModel(args)
    tm.predict()
    ref = pd.read_csv(args.csv_path)
    df_test = pd.read_parquet(tm.path_manager.predict_fpath)
    df_train = pd.read_parquet(tm.path_manager.predict_fpath_with_prefix("train"))
    df_val = pd.read_parquet(tm.path_manager.predict_fpath_with_prefix("val")) 
    assert df_test.shape[1] == 512 + 2
    assert df_train.shape[1] == 512 + 2
    assert df_val.shape[1] == 512 + 2

    assert df_test.shape[0] == ref.query("`0` == 'test'").shape[0]
    assert df_train.shape[0] == ref.query("`0` == 'train'").shape[0]
    assert df_val.shape[0] == ref.query("`0` == 'val'").shape[0]