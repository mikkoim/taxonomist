import pytest
import pandas as pd

from taxonomist.preprocessing import TrainTestSplitArgs, train_test_split

def test_train_test_splits():
    args = TrainTestSplitArgs(
        csv_path = "tests/data/preprocessing/bm_metadata.parquet.gzip",
        target_col = "taxon",
        group_col = "individual",
        n_splits = 5,
        verbose = 1,
        random_state=42,
        shuffle=True,
        out_folder = "test_outputs",
        generate_class_map=True
    )
    train_test_split(args)
    with open("test_outputs/bm_metadata.parquet_5splits_taxon_class_map.txt", "r") as f:
        content = [x.strip() for x in f.readlines()]
    
    assert content == ["Asellus_aquaticus", "Caenis_horaria", "Kageronia_fuscogrisea"]
    df = pd.read_csv("test_outputs/bm_metadata.parquet_5splits_taxon.csv")

    assert df.shape == (10798, 36)
    splits = df.iloc[:, -5:]
    assert (splits.columns == ['0', '1', '2', '3', '4']).all()

    # Only one 'test' per row/fold
    assert ((splits == 'test').sum(axis=1) == 1).all()
    