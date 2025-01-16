import pytest
from taxonomist.taxonomist_model import TaxonomistModel

@pytest.mark.usefixtures("args_basic_train")
def test_training(args_basic_train):
    args = args_basic_train
    tm = TaxonomistModel(args)
    tm.train()