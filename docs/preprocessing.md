## Testing the dataset

```bash
python scripts/preprocessing/test_dataset_config.py \
    --data_folder "$TMPDIR/IDA/" \
    --dataset_config "conf/user_datasets.py" \
    --dataset_name "finbenthic2" \
    --csv_path "data/processed/finbenthic2/01_finbenthic2_processed_5splits_taxon.csv" \
    --label "taxon" \
    --fold 0
```

### Resizing the images

```bash

python scripts/preprocessing/resize_dataset.py \
    --data_folder "$TMPDIR/IDA/" \
    --dataset_config "conf/user_datasets.py" \
    --dataset_name "finbenthic2" \
    --csv_path "data/processed/finbenthic2/01_finbenthic2_processed_5splits_taxon.csv" \
    --label "taxon" \
    --n_folds 5 \
    --out_folder "$TMPDIR/IDA_small" \
    --imsize 224
```