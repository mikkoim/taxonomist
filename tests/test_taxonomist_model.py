import pytest
from taxonomist.taxonomist_model import TaxonomistCheckpoint, TaxonomistModelArguments, PathManager, TaxonomistModel
from pathlib import Path

def test_taxonomist_checkpoint_existing_path():
    existing_ckpt_path = "tests/data/rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last.ckpt"
    checkpoint = TaxonomistCheckpoint(existing_ckpt_path)

    assert checkpoint.is_last()
    assert checkpoint.name == "rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last"
    assert checkpoint.modelname == "rodi_resnet18_f0_250113-1038-a44d"
    assert checkpoint.basename == "rodi_resnet18"
    assert checkpoint.uid == "250113-1038-a44d"
    assert str(checkpoint.folder) == "tests/data"
    
    assert isinstance(checkpoint.ckpt, dict)

def test_taxonomist_checkpoint_nonexisting_path():
    nonexisting_ckpt_path = "tests/data/nonexistent.ckpt"
    with pytest.raises(ValueError) as excinfo:
        checkpoint = TaxonomistCheckpoint(nonexisting_ckpt_path)
    assert("The checkpoint path 'tests/data/nonexistent.ckpt' does not exist" in str(excinfo.value))

def test_taxonomist_checkpoint_nonlast():
    ckpt_path = "tests/data/rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45.ckpt"
    checkpoint = TaxonomistCheckpoint(ckpt_path)

    assert not checkpoint.is_last()
    assert checkpoint.name == "rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45"
    assert checkpoint.modelname == "rodi_resnet18_f0_250113-1038-a44d"
    assert checkpoint.basename == "rodi_resnet18"
    assert checkpoint.uid == "250113-1038-a44d"
    assert str(checkpoint.folder) == "tests/data"

    assert isinstance(checkpoint.ckpt, dict)

def test_validate_arguments_resume():
    pass 

@pytest.mark.usefixtures("args_basic_train")
def test_path_manager_basic_train(args_basic_train):
    args = args_basic_train
    pm = PathManager("training", args)
    assert pm.ckpt is None

    assert pm.modelname.startswith("rodi_resnet18_f0_")
    assert pm.basename == "rodi_resnet18"

    assert isinstance(pm.out_folder, Path)
    assert pm.out_folder.exists()
    assert pm.out_folder.is_dir()
    assert str(pm.out_folder) == "outputs/rodi/rodi_resnet18/f0"

@pytest.mark.usefixtures("args_basic_resume_train")
def test_path_manager_basic_train(args_basic_resume_train):
    args = args_basic_resume_train

    with pytest.raises(ValueError) as excinfo:
        pm = PathManager("training", args)
        assert("A checkpoint path was provided but the checkpoint object is None" in str(excinfo.value))

    ckpt = TaxonomistCheckpoint(args.ckpt_path)
    pm = PathManager("training", args, ckpt)
    assert pm.ckpt is not None

    assert pm.modelname == "rodi_resnet18_f0_250113-1038-a44d"
    assert pm.basename == "rodi_resnet18"

    assert isinstance(pm.out_folder, Path)
    assert pm.out_folder.exists()
    assert pm.out_folder.is_dir()
    assert str(pm.out_folder) == "outputs/rodi/rodi_resnet18/f0"

@pytest.mark.usefixtures("args_basic_predict")
def test_path_manager_basic_predict(args_basic_predict):
    args = args_basic_predict

    with pytest.raises(ValueError) as excinfo:
        pm = PathManager("prediction", args)
        assert("A checkpoint path was provided but the checkpoint object is None" in str(excinfo.value))

    ckpt = TaxonomistCheckpoint(args.ckpt_path)
    pm = PathManager("prediction", args, ckpt)
    assert pm.ckpt is not None

    assert pm.modelname == "rodi_resnet18_f0_250113-1038-a44d"
    assert pm.basename == "rodi_resnet18"

    assert isinstance(pm.out_folder, Path)
    assert pm.out_folder.exists()
    assert pm.out_folder.is_dir()
    assert str(pm.out_folder) == "tests/data/predictions/rodi_none"

    predict_fpath_cls = pm._create_predict_fpath(task="classification")
    predict_fpath_fea = pm._create_predict_fpath(task="feature-extraction")
    assert str(predict_fpath_cls) == "tests/data/predictions/rodi_none/rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last_none.csv"
    assert str(predict_fpath_fea) == "tests/data/predictions/rodi_none/rodi_resnet18_f0_250113-1038-a44d_None.p.gz"

