import pytest
from taxonomist.taxonomist_model import TaxonomistCheckpoint

def test_taxonomist_checkpoint_existing_path():
    existing_ckpt_path = "tests/data/rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last.ckpt"
    checkpoint = TaxonomistCheckpoint(existing_ckpt_path)

    assert checkpoint.is_last()
    assert checkpoint.name == "rodi_resnet18_f0_250113-1038-a44d_epoch18_val-loss1.45_last"
    assert checkpoint.basename == "rodi_resnet18"
    assert checkpoint.uid == "250113-1038-a44d"
    
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
    assert checkpoint.basename == "rodi_resnet18"
    assert checkpoint.uid == "250113-1038-a44d"

    assert isinstance(checkpoint.ckpt, dict)

def test_validate_arguments_resume():
    pass 