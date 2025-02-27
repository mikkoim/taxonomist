import pytest
from taxonomist.taxonomist_model import TaxonomistModel

@pytest.mark.usefixtures("args_basic_predict")
def test_predict(args_basic_predict):
    args = args_basic_predict
    tm = TaxonomistModel(args)
    tm.predict()

@pytest.mark.usefixtures("args_basic_feature_extraction")
def test_predict(args_basic_feature_extraction):
    args = args_basic_feature_extraction
    tm = TaxonomistModel(args)
    tm.predict()

@pytest.mark.usefixtures("args_feature_extraction_pretrained")
def test_predict(args_feature_extraction_pretrained):
    args = args_feature_extraction_pretrained
    tm = TaxonomistModel(args)
    tm.predict()