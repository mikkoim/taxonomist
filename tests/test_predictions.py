import pytest
from taxonomist.taxonomist_model import TaxonomistModel
import pandas as pd

@pytest.mark.usefixtures("args_basic_predict")
def test_predict(args_basic_predict):
    args = args_basic_predict
    tm = TaxonomistModel(args)
    tm.predict()

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