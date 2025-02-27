import pytest
from taxonomist.taxonomist_model import TaxonomistModel

@pytest.mark.usefixtures("args_basic_train")
def test_training(args_basic_train):
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